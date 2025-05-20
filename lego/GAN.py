"""
GAN module, inspired by the CTGAN paper
https://arxiv.org/abs/1907.00503
"""
import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
from tqdm import tqdm
from .data_transformer import DataTransformer
from .base import BaseSynthesizer, random_state


class Discriminator(Module):
    """
    Discriminator for the GAN.

    Args:
        input_dim (int): Size of the input data.
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers.
            A Linear Layer will be created for each one of the values provided.
        pac (int): Number of samples to group together. Defaults to 10.
    """

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """
        Apply the Discriminator to the `input_`.
        """
        assert input_.shape[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))

class Residual(Module):
    """
    Residual layer for the GAN.
    """

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(Module):
    """
    Generator for the GAN.

    Args:
        embedding_dim (int): Size of the random sample passed to the Generator.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals.
            A Residual Layer will be created for each one of the values provided.
        data_dim (int): Size of the output data.
    """

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input_):
        return self.seq(input_)

class GAN(BaseSynthesizer):
    """
    GAN Synthesizer.

    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original GAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
    """

    def __init__(
        self,
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        batch_size=500,
        discriminator_steps=1,
        log_frequency=True,
        verbose=False,
        epochs=300,
        pac=10,
        cuda=True,
        type='continuous',
        folder='LEGO-GAN',
    ):
        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self.loss_values = None
        self.type = type
        self.root_folder = folder
        self.type_folder = os.path.join(self.root_folder, self.type)
        self.samples_folder = os.path.join(self.type_folder, 'samples')
        self.model_folder = os.path.join(self.type_folder, 'models')
        os.makedirs(self.root_folder, exist_ok=True)
        os.makedirs(self.type_folder, exist_ok=True)
        os.makedirs(self.samples_folder, exist_ok=True)
        os.makedirs(self.model_folder, exist_ok=True)

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        for _ in range(10):
            transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError('gumbel_softmax returning NaN.')

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)


    def save(self, filepath):
        """Save the trained model to a file."""
        with open(filepath, 'wb') as f:
            joblib.dump({
                'generator': self._generator,
                'transformer': self._transformer,
                'embedding_dim': self._embedding_dim,
                'generator_dim': self._generator_dim,
                'discriminator_dim': self._discriminator_dim,
                'device': self._device
            }, f)

    def load(self, filepath):
        """Load a trained model from a file."""
        with open(filepath, 'rb') as f:
            state = joblib.load(f)
            self._generator = state['generator']
            self._transformer = state['transformer']
            self._embedding_dim = state['embedding_dim']
            self._generator_dim = state['generator_dim']
            self._discriminator_dim = state['discriminator_dim']
            self._device = state['device']
            self._generator.to(self._device)
        return self

    def plot_losses(self, filename='gan_loss_plot.png'):
        """
        Plot the generator and discriminator losses.
        """
        import matplotlib.pyplot as plt

        # Plot the loss values over epochs
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_values['Epoch'], 
                 self.loss_values['Generator Loss'],
                 label='Generator Loss')
        plt.plot(self.loss_values['Epoch'],
                 self.loss_values['Discriminator Loss'],
                 label='Discriminator Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GAN Training Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.root_folder, filename))
        plt.close()

    @random_state
    def fit(self, train_data, discrete_columns=(), epochs=None):
        """
        Fit the GAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)
        epochs = self._epochs
        train_data = self._transformer.transform(train_data)
        print(f"Transformed data shape {train_data.shape} \n")
        train_data = torch.from_numpy(train_data.astype('float32')).to(self._device)
        print(f" Device: {self._device} \n")
        data_len = len(train_data)
        data_dim = self._transformer.output_dimensions

        self._generator = Generator(
            self._embedding_dim , self._generator_dim, data_dim
        ).to(self._device)

        discriminator = Discriminator(
            data_dim , self._discriminator_dim, pac=self.pac
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        optimizerD = optim.Adam(
            discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])

        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0))

        steps_per_epoch = max(data_len // self._batch_size, 1)

        for i in epoch_iterator:
            for id_ in range(steps_per_epoch):
                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)
                    idx = torch.randint(0, data_len, (self._batch_size,))
                    real = train_data[idx]
                    fakez = torch.normal(mean=mean, std=std)
                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)
                    # Train with real data
                    y_real = discriminator(real)
                    
                    # Train with fake data
                    y_fake = discriminator(fakeact)

                    # Compute WGAN loss with gradient penalty
                    pen = discriminator.calc_gradient_penalty(
                        real, fakeact, self._device, self.pac
                    )
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake)) + pen

                    optimizerD.zero_grad(set_to_none=False)
                    loss_d.backward()
                    optimizerD.step()

                # Train Generator
                fakez = torch.normal(mean=mean, std=std)
                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                y_fake = discriminator(fakeact)
                loss_g = -torch.mean(y_fake)

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()


            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss],
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(
                    drop=True
                )
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss)
                )
            # Save model and generate samples every 50 epochs
            if (i + 1) % 50 == 0:
                
                # Save model

                model_path = f'{self.model_folder}/GAN_{self.type}_model_checkpoint_epoch_{i+1}.pkl'

                self.save(model_path)
                print(f"GAN model saved at epoch {i+1} to {model_path}")
                
                # Generate 100k samples

                samples_path = f'{self.samples_folder}/GAN_{self.type}_samples_epoch_{i+1}.csv'
                samples = self.sample(500000)
                
                # Save samples (assuming they're a DataFrame or can be converted to one)
                if isinstance(samples, np.ndarray):
                    pd.DataFrame(samples).to_csv(samples_path, index=False)
                else:
                    samples.to_csv(samples_path, index=False)
                print(f"Generated 100k samples at epoch {i+1}, saved to {samples_path}")
        
        # Plot losses at the end of training
        self.plot_losses()

    @random_state
    def sample(self, samples):
        """
        Sample data similar to the training data.

        Args:
            samples (int): Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        steps = samples // self._batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]

        return self._transformer.inverse_transform(data)

    def set_device(self, device):
        """
        Set the `device` to be used ('GPU' or 'CPU).
        """
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
