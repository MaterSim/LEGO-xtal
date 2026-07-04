"""Factorized VAE with a Phase-A Ti-skeleton interface for the two-stage Ti/O workflow.

The training architecture retains the current global, Ti-skeleton, Ti-coordinate,
O-skeleton, and O-coordinate reconstruction heads.  Phase-A generation uses only
space group and Ti Wyckoff skeleton; lattice and Ti free coordinates are built by
the symmetry-constrained chemistry builder in ``1_train_sample.py``.
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import Linear, Module, ModuleList, Parameter, ReLU, Sequential
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from pyxtal.symmetry import Group

from .data_transformer import DataTransformer
from .base import BaseSynthesizer, random_state


class MLP(Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers, dim = [], int(input_dim)
        for width in hidden_dims:
            layers.extend([Linear(dim, int(width)), ReLU()])
            dim = int(width)
        layers.append(Linear(dim, int(output_dim)))
        self.net = Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Encoder(Module):
    def __init__(self, data_dim, compress_dims, embedding_dim):
        super().__init__()
        layers, dim = [], int(data_dim)
        for width in compress_dims:
            layers.extend([Linear(dim, int(width)), ReLU()])
            dim = int(width)
        self.body = Sequential(*layers)
        self.fc_mu = Linear(dim, int(embedding_dim))
        self.fc_logvar = Linear(dim, int(embedding_dim))

    def forward(self, x):
        h = self.body(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class ConditionalDecoder(Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.net = MLP(input_dim, hidden_dims, output_dim)
        self.sigma = Parameter(torch.ones(int(output_dim)) * 0.1)

    def forward(self, x):
        return self.net(x), self.sigma


def _activate_transformed(logits, output_info_list, temperature=1.0, hard=False,
                          categorical_masks=None):
    st, outputs = 0, []
    for column_info in output_info_list:
        for span_info in column_info:
            ed = st + span_info.dim
            span = logits[:, st:ed]
            if span_info.activation_fn != "softmax":
                outputs.append(torch.tanh(span))
            else:
                if categorical_masks is not None and (st, ed) in categorical_masks:
                    mask = categorical_masks[(st, ed)]
                    if mask.shape != span.shape:
                        raise ValueError("Categorical mask shape mismatch.")
                    span = span.masked_fill(~mask, -torch.inf)
                scale = float(temperature) if temperature and temperature > 0 else 1.0
                probs = torch.softmax(span / scale, dim=-1)
                if hard:
                    ids = torch.multinomial(probs, 1).squeeze(1)
                    probs = torch.nn.functional.one_hot(ids, span_info.dim).float()
                outputs.append(probs)
            st = ed
    return torch.cat(outputs, dim=1)


def _get_discrete_span_and_categories(transformer, column_name):
    st = 0
    for info in transformer._column_transform_info_list:
        ed = st + info.output_dimensions
        if info.column_name == column_name:
            if info.column_type != "discrete":
                raise ValueError(f"Column {column_name!r} is not discrete.")
            categories = list(info.transform.dummies)
            return st, ed, categories
        st = ed
    raise KeyError(column_name)


def _get_column_span(transformer, column_name):
    st = 0
    for info in transformer._column_transform_info_list:
        ed = st + info.output_dimensions
        if info.column_name == column_name:
            return st, ed
        st = ed
    raise KeyError(column_name)


def _parse_wp_token(token):
    return [int(value) for value in str(token).strip().split("|")]


def _block_loss(recon_x, x, sigmas, output_info, factor, include=None, row_weight=None):
    st = 0
    per_row = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
    used = False
    for column_info in output_info:
        for span_info in column_info:
            ed = st + span_info.dim
            selected = include is None or include(st, ed)
            if selected:
                used = True
                if span_info.activation_fn != "softmax":
                    std = sigmas[st:ed]
                    residual = x[:, st:ed] - torch.tanh(recon_x[:, st:ed])
                    per_row += ((residual ** 2) / (2 * std ** 2)).sum(dim=1)
                    per_row += torch.log(std).sum()
                else:
                    per_row += cross_entropy(
                        recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1),
                        reduction="none"
                    )
            st = ed
    if not used:
        return recon_x.sum() * 0.0
    if row_weight is None:
        row_weight = torch.ones_like(per_row)
    return (per_row * row_weight).sum() * factor / row_weight.sum().clamp_min(1.0)


def _merge_skeleton_and_coordinates(skeleton_x, coordinate_x, span):
    st, ed = span
    output = coordinate_x.clone()
    output[:, st:ed] = skeleton_x[:, st:ed]
    return output


class FactorizedVAE(BaseSynthesizer):
    """Factorized binary VAE v49 with the Phase-A Ti-skeleton sampler."""

    def __init__(self, embedding_dim=128, compress_dims=(512, 512),
                 decompress_dims=(512, 512), context_dim=128, l2scale=1e-5,
                 batch_size=500, epochs=300, loss_factor=2.0,
                 global_loss_weight=1.0, si_loss_weight=1.0, o_loss_weight=1.0,
                 kl_weight=1.0, kl_warmup_epochs=0,
                 predicted_context_start=0.0, predicted_context_end=0.8,
                 o_noise_dim=32, o_noise_scale=1.0, cuda=True, verbose=False,
                 folder="LEGO-FactorizedVAE"):
        self.embedding_dim = int(embedding_dim)
        self.compress_dims = tuple(compress_dims)
        self.decompress_dims = tuple(decompress_dims)
        self.context_dim = int(context_dim)
        self.l2scale = float(l2scale)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.loss_factor = float(loss_factor)
        self.global_loss_weight = float(global_loss_weight)
        self.si_loss_weight = float(si_loss_weight)
        self.o_loss_weight = float(o_loss_weight)
        self.kl_weight = float(kl_weight)
        self.kl_warmup_epochs = int(kl_warmup_epochs)
        self.predicted_context_start = float(predicted_context_start)
        self.predicted_context_end = float(predicted_context_end)
        self.o_noise_dim = int(o_noise_dim)
        self.o_noise_scale = float(o_noise_scale)
        self.verbose = bool(verbose)
        self.root_folder = folder
        device = "cpu" if not cuda or not torch.cuda.is_available() else (
            cuda if isinstance(cuda, str) else "cuda"
        )
        self._device = torch.device(device)
        self.model_folder = os.path.join(folder, "models")
        os.makedirs(self.model_folder, exist_ok=True)

    def _context_probability(self, epoch):
        if self.epochs <= 1:
            return self.predicted_context_end
        fraction = epoch / float(self.epochs - 1)
        return self.predicted_context_start + fraction * (
            self.predicted_context_end - self.predicted_context_start
        )

    @staticmethod
    def _mix_context(true_x, predicted_x, probability):
        if probability <= 0:
            return true_x
        if probability >= 1:
            return predicted_x
        mask = torch.rand(true_x.size(0), 1, device=true_x.device) < probability
        return torch.where(mask, predicted_x, true_x)

    def _build_models(self, global_dim, si_dim, o_dim):
        total_dim = global_dim + si_dim + o_dim
        self.encoder = Encoder(total_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.global_decoder = ConditionalDecoder(self.embedding_dim, self.decompress_dims, global_dim).to(self._device)
        self.global_context_encoder = MLP(global_dim, (self.context_dim,), self.context_dim).to(self._device)
        self.si_skeleton_decoder = ConditionalDecoder(
            self.embedding_dim + self.context_dim, self.decompress_dims, si_dim
        ).to(self._device)
        self.si_skeleton_context_encoder = MLP(si_dim, (self.context_dim,), self.context_dim).to(self._device)
        self.si_coordinate_decoders = ModuleList([
            ConditionalDecoder(self.embedding_dim + 3 * self.context_dim,
                               self.decompress_dims, si_dim)
            for _ in range(self.n_si_sites)
        ]).to(self._device)
        self.si_context_encoder = MLP(si_dim, (self.context_dim,), self.context_dim).to(self._device)
        self.o_si_latent_encoder = MLP(si_dim, (self.context_dim, self.context_dim), self.embedding_dim).to(self._device)
        self.o_noise_encoder = MLP(self.o_noise_dim, (self.context_dim,), self.embedding_dim).to(self._device)
        self.o_skeleton_decoder = ConditionalDecoder(
            self.embedding_dim + 2 * self.context_dim, self.decompress_dims, o_dim
        ).to(self._device)
        self.o_skeleton_context_encoder = MLP(o_dim, (self.context_dim,), self.context_dim).to(self._device)
        self.o_coordinate_decoder = ConditionalDecoder(
            self.embedding_dim + 3 * self.context_dim, self.decompress_dims, o_dim
        ).to(self._device)

    def _all_modules(self):
        return [self.encoder, self.global_decoder, self.global_context_encoder,
                self.si_skeleton_decoder, self.si_skeleton_context_encoder,
                self.si_coordinate_decoders, self.si_context_encoder,
                self.o_si_latent_encoder, self.o_noise_encoder,
                self.o_skeleton_decoder, self.o_skeleton_context_encoder,
                self.o_coordinate_decoder]

    def save(self, filepath):
        names = ["encoder", "global_decoder", "global_context_encoder",
                 "si_skeleton_decoder", "si_skeleton_context_encoder",
                 "si_coordinate_decoders", "si_context_encoder",
                 "o_si_latent_encoder", "o_noise_encoder", "o_skeleton_decoder",
                 "o_skeleton_context_encoder", "o_coordinate_decoder",
                 "global_transformer", "si_transformer", "o_transformer",
                 "si_skeleton_span", "o_skeleton_span", "si_site_spans",
                 "n_si_sites", "si_active_lookup"]
        state = {name: getattr(self, name) for name in names}
        state["device"] = str(self._device)
        state["config"] = {
            "embedding_dim": self.embedding_dim,
            "context_dim": self.context_dim,
            "phase_a_generation": "spg_plus_ti_skeleton_only",
            "direct_lattice_and_ti_coordinate_heads": "training_only",
        }
        joblib.dump(state, filepath)

    def load(self, filepath):
        state = joblib.load(filepath)
        for name, value in state.items():
            if name not in {"device", "config"}:
                setattr(self, name, value)
        self._device = torch.device(state.get("device", "cpu"))
        for module in self._all_modules():
            module.to(self._device)
        if isinstance(self.si_active_lookup, torch.Tensor):
            self.si_active_lookup = self.si_active_lookup.to(self._device)
        return self

    def set_device(self, device):
        self._device = torch.device(device)
        for module in self._all_modules():
            module.to(self._device)
        self.si_active_lookup = self.si_active_lookup.to(self._device)
        return self

    @random_state
    def fit(self, global_data, si_data, o_data, global_discrete_columns=(),
            si_discrete_columns=(), o_discrete_columns=()):
        if not (len(global_data) == len(si_data) == len(o_data)):
            raise ValueError("Global, Ti, and O blocks must have equal row counts.")
        self.global_transformer = DataTransformer()
        self.si_transformer = DataTransformer()
        self.o_transformer = DataTransformer()
        self.global_transformer.fit(global_data, global_discrete_columns)
        self.si_transformer.fit(si_data, si_discrete_columns)
        self.o_transformer.fit(o_data, o_discrete_columns)
        gx = self.global_transformer.transform(global_data).astype("float32")
        sx = self.si_transformer.transform(si_data).astype("float32")
        ox = self.o_transformer.transform(o_data).astype("float32")

        s0, s1, si_categories = _get_discrete_span_and_categories(
            self.si_transformer, "si_skeleton_token"
        )
        o0, o1, _ = _get_discrete_span_and_categories(
            self.o_transformer, "o_skeleton_token"
        )
        self.si_skeleton_span, self.o_skeleton_span = (s0, s1), (o0, o1)
        parsed = [_parse_wp_token(token) for token in si_categories]
        self.n_si_sites = max(len(token) for token in parsed)
        self.si_site_spans = [
            tuple(_get_column_span(self.si_transformer, f"si_u{axis}_{site}")
                  for axis in range(3))
            for site in range(self.n_si_sites)
        ]
        active = np.zeros((len(parsed), self.n_si_sites), dtype=np.float32)
        for category, token in enumerate(parsed):
            for site, wp in enumerate(token):
                active[category, site] = float(wp >= 0)
        self.si_active_lookup = torch.from_numpy(active).to(self._device)
        self._build_models(gx.shape[1], sx.shape[1], ox.shape[1])

        dataset = TensorDataset(torch.from_numpy(gx), torch.from_numpy(sx), torch.from_numpy(ox))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        parameters = [p for module in self._all_modules() for p in module.parameters()]
        optimizer = Adam(parameters, weight_decay=self.l2scale)
        scaler = torch.amp.GradScaler("cuda", enabled=self._device.type == "cuda")
        self.loss_values = []

        for epoch in range(self.epochs):
            running = {k: 0.0 for k in ("global", "si_skeleton", "si_coordinates",
                                        "o_skeleton", "o_coordinates", "kl")}
            count = 0
            context_probability = self._context_probability(epoch)
            kw = self.kl_weight
            if self.kl_warmup_epochs > 0:
                kw *= min(1.0, (epoch + 1) / self.kl_warmup_epochs)
            for gb, sb, ob in loader:
                gb, sb, ob = gb.to(self._device), sb.to(self._device), ob.to(self._device)
                full = torch.cat([gb, sb, ob], dim=1)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=self._device.type == "cuda"):
                    mu, std, logvar = self.encoder(full)
                    z = mu + torch.randn_like(std) * std
                    graw, gsigma = self.global_decoder(z)
                    gpred = _activate_transformed(graw, self.global_transformer.output_info_list)
                    gctx = self.global_context_encoder(self._mix_context(gb, gpred, context_probability))
                    gteacher = self.global_context_encoder(gb)

                    sraw, ssigma = self.si_skeleton_decoder(torch.cat([z, gctx], dim=1))
                    spred_full = _activate_transformed(sraw, self.si_transformer.output_info_list)
                    strue = torch.zeros_like(sb); strue[:, s0:s1] = sb[:, s0:s1]
                    spred = torch.zeros_like(sb); spred[:, s0:s1] = spred_full[:, s0:s1]
                    skel_ctx = self.si_skeleton_context_encoder(strue)
                    partial = strue.clone()
                    scoord = torch.zeros_like(sb)
                    scoord_loss = sb.sum() * 0.0
                    active_rows = self.si_active_lookup[torch.argmax(sb[:, s0:s1], dim=1)]
                    for site, decoder in enumerate(self.si_coordinate_decoders):
                        pctx = self.si_context_encoder(partial)
                        raw, sigma = decoder(torch.cat([z, gteacher, skel_ctx, pctx], dim=1))
                        pred = _activate_transformed(raw, self.si_transformer.output_info_list)
                        spans = self.si_site_spans[site]
                        for st, ed in spans:
                            scoord[:, st:ed] = pred[:, st:ed]
                            partial[:, st:ed] = sb[:, st:ed]
                        scoord_loss += _block_loss(
                            raw, sb, sigma, self.si_transformer.output_info_list,
                            self.loss_factor, include=lambda st, ed, spans=spans: (st, ed) in spans,
                            row_weight=active_rows[:, site]
                        )
                    sfull = _merge_skeleton_and_coordinates(spred_full, scoord, (s0, s1))
                    sctx_input = self._mix_context(sb, sfull, context_probability)
                    sctx = self.si_context_encoder(sctx_input)
                    eps = torch.randn(z.size(0), self.o_noise_dim, device=z.device)
                    zo = z + self.o_si_latent_encoder(sctx_input) + self.o_noise_scale * self.o_noise_encoder(eps)
                    oraw_s, osigma_s = self.o_skeleton_decoder(torch.cat([zo, gctx, sctx], dim=1))
                    opred_s = _activate_transformed(oraw_s, self.o_transformer.output_info_list)
                    otrue = torch.zeros_like(ob); otrue[:, o0:o1] = ob[:, o0:o1]
                    opred = torch.zeros_like(ob); opred[:, o0:o1] = opred_s[:, o0:o1]
                    osctx = self.o_skeleton_context_encoder(self._mix_context(otrue, opred, context_probability))
                    oraw_c, osigma_c = self.o_coordinate_decoder(torch.cat([zo, gctx, sctx, osctx], dim=1))

                    gl = _block_loss(graw, gb, gsigma, self.global_transformer.output_info_list, self.loss_factor)
                    ssl = _block_loss(sraw, sb, ssigma, self.si_transformer.output_info_list,
                                      self.loss_factor, include=lambda st, ed: st == s0 and ed == s1)
                    osl = _block_loss(oraw_s, ob, osigma_s, self.o_transformer.output_info_list,
                                      self.loss_factor, include=lambda st, ed: st == o0 and ed == o1)
                    ocl = _block_loss(oraw_c, ob, osigma_c, self.o_transformer.output_info_list,
                                      self.loss_factor, include=lambda st, ed: not (st == o0 and ed == o1))
                    kl = -0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp()) / full.size(0)
                    loss = self.global_loss_weight * gl + self.si_loss_weight * (ssl + scoord_loss) + \
                           self.o_loss_weight * (osl + ocl) + kw * kl
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                scaler.step(optimizer); scaler.update()
                for decoder in [self.global_decoder, self.si_skeleton_decoder,
                                *list(self.si_coordinate_decoders), self.o_skeleton_decoder,
                                self.o_coordinate_decoder]:
                    decoder.sigma.data.clamp_(0.01, 1.0)
                bsz = full.size(0); count += bsz
                for key, value in (("global", gl), ("si_skeleton", ssl),
                                   ("si_coordinates", scoord_loss), ("o_skeleton", osl),
                                   ("o_coordinates", ocl), ("kl", kl)):
                    running[key] += float(value.detach()) * bsz
            record = {f"{key}_loss": value / count for key, value in running.items()}
            record.update(epoch=epoch + 1, predicted_context_probability=context_probability,
                          kl_weight=kw)
            self.loss_values.append(record)
            if self.verbose and (epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == self.epochs):
                total = self.global_loss_weight * record["global_loss"] + \
                        self.si_loss_weight * (record["si_skeleton_loss"] + record["si_coordinates_loss"]) + \
                        self.o_loss_weight * (record["o_skeleton_loss"] + record["o_coordinates_loss"]) + \
                        kw * record["kl_loss"]
                print(f"Epoch {epoch + 1:4d}/{self.epochs} | loss {total:.3f} | ctx {context_probability:.2f}")
        pd.DataFrame(self.loss_values).to_csv(os.path.join(self.root_folder, "factorized_loss.csv"), index=False)

    @random_state
    def sample_ti_skeletons(self, samples, temperature=1.0, hard=True,
                            composition_ratio=(1, 2), max_independent_sites=None):
        """Sample only space group and Ti Wyckoff skeleton for Phase A."""
        center_coeff, sublattice_coeff = map(int, composition_ratio)
        if min(center_coeff, sublattice_coeff) <= 0:
            raise ValueError("composition_ratio must contain positive integers.")
        for module in self._all_modules():
            module.eval()
        sp0, sp1, sp_categories = _get_discrete_span_and_categories(self.global_transformer, "spg")
        s0, s1, si_categories = _get_discrete_span_and_categories(self.si_transformer, "si_skeleton_token")
        _, _, o_categories = _get_discrete_span_and_categories(self.o_transformer, "o_skeleton_token")
        parsed_si = [_parse_wp_token(x) for x in si_categories]
        parsed_o = [_parse_wp_token(x) for x in o_categories]
        si_site_counts = [sum(w >= 0 for w in x) for x in parsed_si]
        o_site_counts = [sum(w >= 0 for w in x) for x in parsed_o]
        groups = {}
        output_spg, output_tokens, output_z, valid_all = [], [], [], []
        stats = {"rows": int(samples), "invalid_space_group": 0,
                 "no_compatible_si_skeleton": 0}
        generated = 0
        with torch.no_grad():
            while generated < samples:
                n = min(self.batch_size, samples - generated)
                z = torch.randn(n, self.embedding_dim, device=self._device)
                graw, _ = self.global_decoder(z)
                gx = _activate_transformed(graw, self.global_transformer.output_info_list,
                                           temperature=temperature, hard=hard)
                gctx = self.global_context_encoder(gx)
                sraw, _ = self.si_skeleton_decoder(torch.cat([z, gctx], dim=1))
                sp_ids = torch.argmax(gx[:, sp0:sp1], dim=1).cpu().numpy()
                allowed = torch.zeros((n, len(si_categories)), dtype=torch.bool, device=self._device)
                row_valid = np.ones(n, dtype=bool)
                realized_spg = []
                for row, category in enumerate(sp_ids):
                    try:
                        spg = int(round(float(sp_categories[int(category)])))
                        group = groups.setdefault(spg, Group(spg))
                        mult = [int(group[i].multiplicity) for i in range(len(group))]
                    except Exception:
                        spg, mult = -1, []
                    realized_spg.append(spg)
                    if not 1 <= spg <= 230:
                        row_valid[row] = False; stats["invalid_space_group"] += 1
                        allowed[row, 0] = True; continue
                    for sid, swps in enumerate(parsed_si):
                        occupied = [w for w in swps if w >= 0]
                        if not occupied or any(w >= len(mult) for w in occupied):
                            continue
                        n_ti = sum(mult[w] for w in occupied)
                        for oid, owps in enumerate(parsed_o):
                            oocc = [w for w in owps if w >= 0]
                            if not oocc or any(w >= len(mult) for w in oocc):
                                continue
                            if max_independent_sites is not None and \
                               si_site_counts[sid] + o_site_counts[oid] > int(max_independent_sites):
                                continue
                            if center_coeff * sum(mult[w] for w in oocc) == sublattice_coeff * n_ti:
                                allowed[row, sid] = True; break
                    if not bool(allowed[row].any()):
                        row_valid[row] = False; stats["no_compatible_si_skeleton"] += 1
                        allowed[row, 0] = True
                sx = _activate_transformed(sraw, self.si_transformer.output_info_list,
                                           temperature=temperature, hard=hard,
                                           categorical_masks={(s0, s1): allowed})
                sid = torch.argmax(sx[:, s0:s1], dim=1).cpu().numpy()
                output_spg.extend(realized_spg)
                output_tokens.extend(str(si_categories[int(i)]) for i in sid)
                output_z.append(z.cpu().numpy())
                valid_all.extend(row_valid.tolist())
                generated += n
        return {"z": np.concatenate(output_z, axis=0)[:samples],
                "spg": np.asarray(output_spg[:samples], dtype=int),
                "si_skeleton_token": np.asarray(output_tokens[:samples], dtype=object),
                "valid_mask": np.asarray(valid_all[:samples], dtype=bool),
                "stats": stats}

