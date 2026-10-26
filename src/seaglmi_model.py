"""Original single SEAGLMI network definitions, extracted without changing their AST.
Optional historical branches are retained for state-dictionary compatibility.
The screened comparison uses only the strict/standard settings in protocols/SCOPE.md.
This module is a model definition, not a new training or ensemble entry point.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
D=64
MI_MAX=28
N_FEAT=14
N_NBR=7

class MiEncoder(nn.Module):
    def __init__(self, d=D):
        super().__init__()
        self.pos = nn.Embedding(MI_MAX, 16)
        self.proj = nn.Linear(640 + 16 + 1, d)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, batch_first=True), 1)

    def forward(self, tok, mask):  # tok [B, L, 640], mask [B, L] True = real
        B, L, _ = tok.shape
        pos = torch.arange(L, device=tok.device).clamp(max=MI_MAX - 1)
        seed = ((pos >= 1) & (pos <= 7)).float()[None, :, None].expand(B, L, 1)
        x = self.proj(torch.cat([tok, self.pos(pos)[None].expand(B, L, -1), seed], -1))
        return self.tf(x, src_key_padding_mask=~mask)

class SiteEncoder(nn.Module):
    def __init__(self, d=D, pairmap=True, biophys=True):
        super().__init__()
        self.pairmap, self.biophys = pairmap, biophys
        self.emb = nn.Embedding(5, 16)
        self.conv = nn.Sequential(nn.Conv1d(16, d, 7, padding=3), nn.ReLU(), nn.Conv1d(d, d, 5, padding=2), nn.ReLU())
        self.map = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
                                 nn.AdaptiveMaxPool2d((4, 6)), nn.Flatten(), nn.Linear(8 * 24, d), nn.ReLU())
        self.num = nn.Sequential(nn.Linear(N_FEAT, d), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU())

    def forward(self, win, feat, mi_tok, mi_mask):
        # win [S, WIN] long, feat [S, F], mi_tok [S, L, d] (miRNA tokens of each site's pair)
        w = self.conv(self.emb(win).transpose(1, 2)).transpose(1, 2)            # S x WIN x d
        a = torch.einsum("sld,swd->slw", mi_tok, w) / D ** 0.5
        a = a.masked_fill(~mi_mask[:, :, None], 0.0)
        pm = self.map(a[:, None]) if self.pairmap else torch.zeros(len(win), D, device=win.device)
        nf = self.num(feat) if self.biophys else torch.zeros(len(win), D, device=win.device)
        return self.out(torch.cat([pm, nf], -1))

class PLR(nn.Module):
    """Periodic-linear-ReLU embedding of numerical features (Gorishniy et al., NeurIPS 2022): each scalar x_j
    becomes ReLU(W_j [sin(2*pi*c_j x_j), cos(2*pi*c_j x_j)]) with learned frequencies c_j ~ N(0, sigma)."""

    def __init__(self, n, k=16, d=8, sigma=1.0):
        super().__init__()
        self.c = nn.Parameter(torch.randn(n, k) * sigma)
        self.w = nn.Parameter(torch.randn(n, 2 * k, d) / (2 * k) ** 0.5)
        self.b = nn.Parameter(torch.zeros(n, d))

    def forward(self, x):                                                       # B x n -> B x (n*d)
        v = 2 * torch.pi * x[..., None] * self.c                                # B x n x k
        h = torch.einsum("bnk,nkd->bnd", torch.cat([torch.sin(v), torch.cos(v)], -1), self.w) + self.b
        return F.relu(h).flatten(1)

def bag_self_attention(layer, sv, site_pair, P):
    """Run one Transformer encoder layer over each pair's site bag (sites of a pair are contiguous)."""
    _, row = torch.unique_consecutive(site_pair, return_inverse=True)          # only pairs that have sites
    counts = torch.bincount(row)
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(len(site_pair), device=sv.device) - start[row]
    kmax = int(counts.max())
    pad = sv.new_zeros(len(counts), kmax, sv.shape[1])
    pad[row, pos] = sv
    mask = torch.arange(kmax, device=sv.device)[None, :] >= counts[:, None]      # True = padding
    out = layer(pad, src_key_padding_mask=mask)
    return out[row, pos]

class SEAGLMI2(nn.Module):
    def __init__(self, n_agg, pairmap=True, biophys=True, use_global=True, attn=True, use_agg=True, use_sites=True,
                 deg_prior=False, nbr=False, expr=False, coop=False, additive=False, plr=False, site_sa=False):
        super().__init__()
        self.deg_prior, self.nbr, self.expr, self.coop = deg_prior, nbr, expr, coop
        self.additive, self.plr, self.site_sa = additive, plr, site_sa
        if site_sa:  # sites of a pair attend to each other (cooperativity, competition) before pooling
            self.sa = nn.TransformerEncoderLayer(D, 4, 128, dropout=0.1, batch_first=True)
        self.use_global, self.attn, self.use_agg, self.use_sites = use_global, attn, use_agg, use_sites
        self.mi = MiEncoder()
        self.site = SiteEncoder(pairmap=pairmap, biophys=biophys)
        self.gate = nn.Sequential(nn.Linear(D, D), nn.Tanh(), nn.Linear(D, 1))
        self.glob_m = nn.Linear(640, 32)
        self.glob_l = nn.Linear(640, 32)
        n_ctx = (2 if deg_prior else 0) + (N_NBR if nbr else 0) + (5 if expr else 0)
        n_in = 2 * D + n_agg + (64 if use_global else 0) + (3 if coop else 0) + (0 if additive else n_ctx)
        if plr:  # numerical pair-level inputs (site summaries, cooperative sites, context) through PLR embeddings
            n_tab = n_agg + (3 if coop else 0) + n_ctx
            self.plr_emb = PLR(n_tab)
            n_in = 2 * D + (64 if use_global else 0) + n_tab * 8
        self.head = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
        if additive:  # score = site evidence + context prior; degree term monotone non-decreasing
            self.deg_w = nn.Parameter(torch.zeros(2))
            n_rest = n_ctx - (2 if deg_prior else 0)
            self.ctx_head = nn.Sequential(nn.Linear(n_rest, 16), nn.ReLU(), nn.Linear(16, 1)) if n_rest else None

    def forward(self, b):
        mt = self.mi(b["mi_tok"], b["mi_mask"])                                 # P x L x d
        S = len(b["site_pair"]) if self.use_sites else 0
        if S:
            sv = self.site(b["win"], b["feat"], mt[b["site_pair"]], b["mi_mask"][b["site_pair"]])
            if self.site_sa:
                sv = bag_self_attention(self.sa, sv, b["site_pair"], mt.shape[0])
            g = self.gate(sv).squeeze(-1)
        P = mt.shape[0]
        att = torch.zeros(P, D, device=mt.device)
        mx = torch.zeros(P, D, device=mt.device)
        w_site = torch.zeros(S, device=mt.device)
        if S:
            gmax = torch.full((P,), -1e9, device=mt.device).scatter_reduce(0, b["site_pair"], g, "amax")
            e = torch.exp(g - gmax[b["site_pair"]])
            den = torch.zeros(P, device=mt.device).index_add(0, b["site_pair"], e)
            w_site = e / den[b["site_pair"]]
            att = att.index_add(0, b["site_pair"], w_site[:, None] * sv)
            mx = mx.scatter_reduce(0, b["site_pair"][:, None].expand(-1, D), sv, "amax", include_self=True)
        pooled = att if self.attn else mx
        parts = [pooled, mx, b["agg"] if self.use_agg else torch.zeros_like(b["agg"])]
        if self.use_global:
            parts += [F.relu(self.glob_m(b["mi_glob"])), F.relu(self.glob_l(b["lnc_glob"]))]
        if self.coop:
            parts.append(b["coop"])
        ctx = ([b["nbr"]] if self.nbr else []) + ([b["expr"]] if self.expr else [])
        if self.plr:
            tab = [b["agg"]] + ([b["coop"]] if self.coop else []) + ([b["deg"]] if self.deg_prior else []) + ctx
            parts = parts[:2] + parts[3:-1 if self.coop else None] + [self.plr_emb(torch.cat(tab, -1))]
            return self.head(torch.cat(parts, -1)).squeeze(-1), w_site
        if not self.additive:
            ctx = ([b["deg"]] if self.deg_prior else []) + ctx
            return self.head(torch.cat(parts + ctx, -1)).squeeze(-1), w_site
        score = self.head(torch.cat(parts, -1)).squeeze(-1)                     # site evidence
        if self.deg_prior:
            score = score + b["deg"] @ F.softplus(self.deg_w)                    # monotone degree prior
        if self.ctx_head is not None:
            score = score + self.ctx_head(torch.cat(ctx, -1)).squeeze(-1)        # neighbour / expression prior
        return score, w_site
