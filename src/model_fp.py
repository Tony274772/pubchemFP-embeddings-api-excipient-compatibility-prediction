"""PubChemFP flat-vector API/excipient compatibility model (no attention)."""

import math
import torch
import torch.nn as nn

from src.model import _projection  # reuse existing helper, do not redefine


class APIExcipientFPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        fp_dim = config.pubchem_fp_dim          # 881
        proj_dim = config.proj_dim              # same proj_dim as the MoLFormer model, for comparability
        hidden = config.fp_proj_hidden_dim

        self.api_proj = _projection(fp_dim, hidden, proj_dim, config.proj_dropout)
        self.exc_proj = _projection(fp_dim, hidden, proj_dim, config.proj_dropout)

        # Single learned placeholder vector for a missing excipient structure
        # (no token dimension here, unlike the MoLFormer model's per-token placeholders).
        self.exc_missing_placeholder = nn.Parameter(torch.randn(1, proj_dim) * 0.01)

        if config.use_descriptors:
            self.api_desc_proj = _projection(
                config.num_descriptors, 32, config.desc_proj_dim, config.desc_dropout
            )
            self.exc_desc_proj = _projection(
                config.num_descriptors, 32, config.desc_proj_dim, config.desc_dropout
            )

        dim_api = proj_dim
        dim_exc = proj_dim + 1
        if config.use_descriptors:
            dim_api = proj_dim + config.desc_proj_dim
            dim_exc = proj_dim + config.desc_proj_dim + 1
        self.dim_api = dim_api

        pair_dim = dim_api + dim_exc + dim_api + dim_api

        self.classifier = nn.Sequential(
            nn.Linear(pair_dim, config.clf_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.clf_dropout_1),
            nn.Linear(config.clf_hidden_dim, config.clf_hidden_dim_2),
            nn.GELU(),
            nn.Dropout(config.clf_dropout_2),
            nn.Linear(config.clf_hidden_dim_2, 1),
        )

        final_linear_layer = self.classifier[-1]
        log_odds = math.log(config.positive_prior / (1.0 - config.positive_prior))
        torch.nn.init.constant_(final_linear_layer.bias, log_odds)

    def forward(self, batch):
        api_fp = batch["api_fp"]           # [B, 881]
        exc_fp = batch["exc_fp"]           # [B, 881]
        exc_avail = batch["exc_available"].unsqueeze(1)  # [B, 1]

        h_api_struct = self.api_proj(api_fp)   # [B, proj_dim]

        exc_struct_real = self.exc_proj(exc_fp)  # [B, proj_dim]
        missing_rows = (exc_avail == 0.0)        # [B, 1]
        h_exc_struct = torch.where(
            missing_rows,
            self.exc_missing_placeholder.expand(exc_struct_real.size(0), -1),
            exc_struct_real,
        )

        if self.config.use_descriptors:
            d_api = self.api_desc_proj(batch["api_desc"])
            d_exc = self.exc_desc_proj(batch["exc_desc"])
            h_api = torch.cat([h_api_struct, d_api], dim=-1)
            h_exc = torch.cat([h_exc_struct, d_exc, exc_avail], dim=-1)
        else:
            h_api = h_api_struct
            h_exc = torch.cat([h_exc_struct, exc_avail], dim=-1)

        h_exc_core = h_exc[:, :self.dim_api]
        interaction = h_api * h_exc_core
        difference = torch.abs(h_api - h_exc_core)

        pair_vec = torch.cat([h_api, h_exc, interaction, difference], dim=-1)
        logits = self.classifier(pair_vec)
        return logits.squeeze(1)
