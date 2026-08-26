"""
VAE module.
"""
import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import (
    Linear,
    Module,
    Parameter,
    ReLU,
    Sequential,
    TransformerEncoderLayer,
    TransformerEncoder,
)
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from .data_transformer import DataTransformer
from .base import BaseSynthesizer, random_state


#random.seed(42)
#np.random.seed(42)
#torch.manual_seed(42)

class Encoder(Module):
    """Encoder for the VAE.

    Args:
        data_dim (int): Dimensions of the data.
        compress_dims (tuple or list of ints): Size of each hidden layer.
        embedding_dim (int): Size of the output vector.
    """

    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input_):
        """Encode the passed `input_`."""
        feature = self.seq(input_)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(Module):
    """Decoder for the TVAE.

    Args:
        embedding_dim (int): Size of the input vector.
        decompress_dims (tuple or list of ints): Size of each hidden layer.
        data_dim (int): Dimensions of the data.
    """

    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input_):
        """Decode the passed `input_`."""
        return self.seq(input_), self.sigma


class PositionalEncoding(Module):
    """Simple positional encoding.

    Supports learned or sinusoidal encodings. Implemented with `batch_first=True`.
    """

    def __init__(self, num_tokens, d_model, learned=True):
        super().__init__()
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.learned = learned
        if learned:
            self.pe = Parameter(torch.zeros(1, num_tokens, d_model))
            torch.nn.init.trunc_normal_(self.pe, std=0.02)
        else:
            pe = torch.zeros(num_tokens, d_model)
            position = torch.arange(0, num_tokens, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [B, T, C]
        T = x.size(1)
        return x + self.pe[:, :T, :]


def _build_token_slices(output_info_list):
    """Build contiguous slices per original column.

    Given the CTGAN-style `output_info_list` which is a list of lists (spans)
    per original column, compute a list of (start, end) indices that cover each
    column's full transformed width.
    """
    slices = []
    st = 0
    for column_info in output_info_list:
        col_start = st
        for span_info in column_info:
            st += span_info.dim
        slices.append((col_start, st))
    return slices


class TransformerEncoderTabular(Module):
    """Transformer-based encoder over column-wise tokens.

    Each original column (possibly composed of multiple spans) is treated as a
    token: the slice is linearly projected to `d_model`, a positional encoding
    is added, and a stack of self-attention layers processes the sequence. The
    pooled representation parameterizes the VAE posterior (mu, logvar).
    """

    def __init__(
        self,
        data_dim,
        output_info_list,
        embedding_dim=128,
        d_model=128,
        nhead=8,
        num_layers=2,
        dropout=0.1,
        use_cls_token=True,
        learned_positional_encoding=True,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.token_slices = _build_token_slices(output_info_list)
        self.num_tokens = len(self.token_slices)
        self.use_cls = use_cls_token

        # Per-token input projections (varying input widths)
        self.token_in = torch.nn.ModuleList(
            [Linear(ed - st, d_model) for st, ed in self.token_slices]
        )

        # Optional [CLS] token
        if self.use_cls:
            self.cls_token = Parameter(torch.zeros(1, 1, d_model))
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional encoding for sequence length (with CLS if used)
        pe_tokens = self.num_tokens + (1 if self.use_cls else 0)
        self.positional_encoding = PositionalEncoding(
            pe_tokens, d_model, learned=learned_positional_encoding
        )

        enc_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True
        )
        self.encoder = TransformerEncoder(enc_layer, num_layers=num_layers)

        self.fc_mu = Linear(d_model, embedding_dim)
        self.fc_logvar = Linear(d_model, embedding_dim)

    def forward(self, input_):
        # input_: [B, data_dim]
        B = input_.size(0)
        tokens = []
        for (st, ed), proj in zip(self.token_slices, self.token_in):
            tok = proj(input_[:, st:ed])  # [B, d_model]
            tokens.append(tok)
        x = torch.stack(tokens, dim=1)  # [B, T, d_model]

        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)  # [B,1,d_model]
            x = torch.cat([cls, x], dim=1)

        x = self.positional_encoding(x)
        x = self.encoder(x)  # [B, T(+1), d_model]

        if self.use_cls:
            feat = x[:, 0, :]
        else:
            feat = x.mean(dim=1)

        mu = self.fc_mu(feat)
        logvar = self.fc_logvar(feat)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class TransformerDecoderTabular(Module):
    """Transformer-based decoder that predicts per-column slices.

    Uses a learned token query for each column. The latent vector is projected
    and added to every token (conditioning). The resulting sequence is processed
    by a Transformer encoder stack (self-attention over tokens). Finally each
    token is projected back to its column slice and concatenated.
    """

    def __init__(
        self,
        embedding_dim,
        output_info_list,
        data_dim,
        d_model=128,
        nhead=8,
        num_layers=2,
        dropout=0.1,
        learned_positional_encoding=True,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.token_slices = _build_token_slices(output_info_list)
        self.num_tokens = len(self.token_slices)
        self.d_model = d_model

        self.latent_proj = Linear(embedding_dim, d_model)
        self.token_queries = Parameter(torch.zeros(1, self.num_tokens, d_model))
        torch.nn.init.trunc_normal_(self.token_queries, std=0.02)

        self.positional_encoding = PositionalEncoding(self.num_tokens, d_model, learned=learned_positional_encoding)

        enc_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True
        )
        self.decoder = TransformerEncoder(enc_layer, num_layers=num_layers)

        # Per-token output heads
        self.token_out = torch.nn.ModuleList(
            [Linear(d_model, ed - st) for st, ed in self.token_slices]
        )

        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input_):
        # input_: latent [B, embedding_dim]
        B = input_.size(0)
        cond = self.latent_proj(input_)  # [B, d_model]
        tokens = self.token_queries.expand(B, -1, -1) + cond.unsqueeze(1)
        tokens = self.positional_encoding(tokens)
        y = self.decoder(tokens)  # [B, T, d_model]

        outs = []
        for h, (st, ed) in zip(self.token_out, self.token_slices):
            outs.append(h(y[:, len(outs), :]))  # [B, slice]
        recon = torch.cat(outs, dim=-1)  # [B, data_dim]
        return recon, self.sigma

def _loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor):
    st = 0
    loss = []
    for column_info in output_info:
        for span_info in column_info:
            ed = st + span_info.dim
            if span_info.activation_fn != 'softmax':
                # Vectorized Gaussian NLL over the whole span
                std = sigmas[st:ed]
                eq = x[:, st:ed] - torch.tanh(recon_x[:, st:ed])
                loss.append(((eq**2) / (2 * (std**2))).sum())
                loss.append(torch.log(std).sum() * x.size()[0])
            else:
                loss.append(
                    cross_entropy(
                        recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'
                    )
                )
            st = ed

    assert st == recon_x.size()[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]


class VAE(BaseSynthesizer):
    """
    Variational Autoencoder (VAE) for tabular data.

    Args:
        embedding_dim (int): Size of the output vector.
        compress_dims (tuple or list of ints): Size of each hidden layer.
        decompress_dims (tuple or list of ints): Size of each hidden layer.
        l2scale (float): L2 regularization factor.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs for training.
        loss_factor (float): Loss factor for the VAE.
        cuda (bool or str): Device to use ('cuda' or 'cpu').
        verbose (bool): Verbosity flag.
        folder (str): Folder to save models and samples.
    """
    def __init__(
        self,
        embedding_dim=128,
        compress_dims=(512, 512),
        decompress_dims=(512, 512),
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
        loss_factor=2,
        cuda=True,
        verbose=False,
        type='discrete',
        folder='LEGO-VAE',
        # Transformer options
        use_attention=False,
        d_model=128,
        nhead=8,
        num_layers=2,
        dropout=0.1,
        learned_positional_encoding=True,
        use_cls_token=True,
        # KL options
        kl_weight=1.0,
        kl_warmup_epochs=0,
    ):
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs
        self.loss_values = pd.DataFrame(columns=['Epoch', 'Batch', 'Loss'])
        self.verbose = verbose
        self.use_attention = use_attention
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.learned_positional_encoding = learned_positional_encoding
        self.use_cls_token = use_cls_token
        self.kl_weight = kl_weight
        self.kl_warmup_epochs = kl_warmup_epochs

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'


        self._device = torch.device(device)
        #store encoded features for last step
        self.full_encoded_mu = []
        self.full_encoded_std = []
        self.type = type
        self.root_folder = folder
        self.type_folder = os.path.join(self.root_folder, self.type)
        self.samples_folder = os.path.join(self.type_folder, 'samples')
        self.model_folder = os.path.join(self.type_folder, 'models')
        os.makedirs(self.root_folder, exist_ok=True)
        os.makedirs(self.type_folder, exist_ok=True)
        os.makedirs(self.samples_folder, exist_ok=True)
        os.makedirs(self.model_folder, exist_ok=True)

    def plot_losses(self, loss_values, filename='loss_plot.png'):
        # Plot the loss values over iterations
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(loss_values.index, loss_values['Loss'], label='Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss over Iterations')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.root_folder, filename))
        plt.close()

    def save(self, filepath):
        """
        Save the trained model to a file.
        """
        with open(filepath, 'wb') as f:
            joblib.dump({
                'encoder': self.encoder,
                'decoder': self.decoder,
                'transformer': self.transformer,
                'device': self._device
            }, f)

    def load(self, filepath):
        """
        Load a trained model from a file.
        """
        with open(filepath, 'rb') as f:
            state = joblib.load(f)
            self.encoder = state['encoder']
            self.decoder = state['decoder']
            self.transformer = state['transformer']
            self._device = state['device']
            self.encoder.to(self._device)
            self.decoder.to(self._device)
        return self


    @random_state
    def fit(self, train_data, discrete_columns=()):
        """
        Fit the TVAE Synthesizer models to the training data.

        Args:
            train_data (np.ndarray or pd.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """

        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)
        os.makedirs("data", exist_ok=True)
        np.save("data/train_data.npy", train_data)
        print("Train Data", np.shape(train_data), '\n', train_data[:10])
        print("Transformed Data shape", np.shape(train_data))

        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        data_dim = self.transformer.output_dimensions
        if self.use_attention:
            self.encoder = TransformerEncoderTabular(
                data_dim=data_dim,
                output_info_list=self.transformer.output_info_list,
                embedding_dim=self.embedding_dim,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                dropout=self.dropout,
                use_cls_token=self.use_cls_token,
                learned_positional_encoding=self.learned_positional_encoding,
            ).to(self._device)
            self.decoder = TransformerDecoderTabular(
                embedding_dim=self.embedding_dim,
                output_info_list=self.transformer.output_info_list,
                data_dim=data_dim,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                dropout=self.dropout,
                learned_positional_encoding=self.learned_positional_encoding,
            ).to(self._device)
        else:
            self.encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
            self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
        optimizerAE = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), weight_decay=self.l2scale
        )

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Batch', 'Loss'])
        iterator = tqdm(range(self.epochs), disable=(not self.verbose))
        if self.verbose:
            iterator_description = 'Loss: {loss:.3f}'
            iterator.set_description(iterator_description.format(loss=0))

        for i in iterator:
            loss_values = []
            batch = []
            # epoch running averages (weighted by batch size)
            epoch_rec_sum = 0.0
            epoch_kl_sum = 0.0
            epoch_count = 0
            #reset full encoded features
            scaler = torch.cuda.amp.GradScaler(enabled=(self._device.type == 'cuda'))
            for id_, data in enumerate(loader):
                optimizerAE.zero_grad(set_to_none=True)
                real = data[0].to(self._device)

                with torch.cuda.amp.autocast(enabled=(self._device.type == 'cuda')):
                    mu, std, logvar = self.encoder(real)
                    eps = torch.randn_like(std)
                    emb = eps * std + mu
                    rec, sigmas = self.decoder(emb)
                    loss_1, loss_2 = _loss_function(
                        rec,
                        real,
                        sigmas,
                        mu,
                        logvar,
                        self.transformer.output_info_list,
                        self.loss_factor,
                    )
                    # KL weighting with optional warmup
                    if self.kl_warmup_epochs and self.kl_warmup_epochs > 0:
                        curr_kl_w = self.kl_weight * min(1.0, (i + 1) / float(self.kl_warmup_epochs))
                    else:
                        curr_kl_w = self.kl_weight
                    loss = loss_1 + curr_kl_w * loss_2

                scaler.scale(loss).backward()
                # Gradient clipping
                scaler.unscale_(optimizerAE)
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()), 1.0
                )
                scaler.step(optimizerAE)
                scaler.update()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

                batch.append(id_)
                loss_values.append(loss.detach().cpu().item())
                bsz = real.size(0)
                epoch_rec_sum += loss_1.detach().cpu().item() * bsz
                epoch_kl_sum += loss_2.detach().cpu().item() * bsz
                epoch_count += bsz

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i] * len(batch),
                'Batch': batch,
                'Loss': loss_values,
            })

            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(
                    drop=True
                )
            else:
                self.loss_values = epoch_loss_df

            if self.verbose and epoch_count > 0:
                avg_rec = epoch_rec_sum / epoch_count
                avg_kl = epoch_kl_sum / epoch_count
                if self.kl_warmup_epochs and self.kl_warmup_epochs > 0:
                    curr_kl_w = self.kl_weight * min(1.0, (i + 1) / float(self.kl_warmup_epochs))
                else:
                    curr_kl_w = self.kl_weight
                avg_total = avg_rec + curr_kl_w * avg_kl
                iterator.set_description(iterator_description.format(loss=avg_total))
                print(
                    f"Epoch {i+1}/{self.epochs} | Avg Recon: {avg_rec:.4f} | Avg KL: {avg_kl:.4f} | KL-w: {curr_kl_w:.3f} | Avg Total: {avg_total:.4f}"
                )
            if (i + 1) % 25 == 0:  
                # Save model
                model_path = f'{self.model_folder}/VAE_{self.type}_model_checkpoint_epoch_{i+1}.pkl'
                self.save(model_path)
                print(f"Model saved at epoch {i+1} to {model_path}")
                
                # Generate 100k samples
                samples_path = f'{self.samples_folder}/VAE_{self.type}_samples_epoch_{i+1}.csv'
                samples = self.sample(100000, temperature=1.5, hard=True)

                # Save samples (assuming they're a DataFrame or can be converted to one)
                if isinstance(samples, np.ndarray):
                    pd.DataFrame(samples).to_csv(samples_path, index=False)
                else:
                    samples.to_csv(samples_path, index=False)
                print(f"Generated 100k samples at epoch {i+1}, saved to {samples_path}")

        self.plot_losses(self.loss_values, filename='epoch_loss_plot.png')

    @random_state
    def sample(self, samples, temperature=1.0, hard=True):
        """
        Sample data similar to the training data.
        
        Args:
            samples (int): Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        self.decoder.eval()
        steps = samples // self.batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)
            fake, sigmas = self.decoder(noise)
            # Apply tanh for continuous spans; categorical via temperature softmax with optional hard sampling
            st = 0
            outs = []
            for column_info in self.transformer.output_info_list:
                for span_info in column_info:
                    ed = st + span_info.dim
                    if span_info.activation_fn != 'softmax':
                        outs.append(torch.tanh(fake[:, st:ed]))
                    else:
                        logits = fake[:, st:ed]
                        if temperature is not None and temperature > 0:
                            probs = torch.softmax(logits / float(temperature), dim=-1)
                        else:
                            probs = torch.softmax(logits, dim=-1)
                        if hard:
                            idx = torch.multinomial(probs, 1).squeeze(1)
                            one_hot = torch.nn.functional.one_hot(idx, num_classes=ed - st).float()
                            outs.append(one_hot)
                        else:
                            outs.append(probs)
                    st = ed
            fake_proc = torch.cat(outs, dim=1)
            data.append(fake_proc.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]
        return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())

    def set_device(self, device):
        """
        Set the `device` to be used ('GPU' or 'CPU).
        """
        self._device = device
        if hasattr(self, 'encoder'):
            self.encoder.to(self._device)
        if hasattr(self, 'decoder'):
            self.decoder.to(self._device)
