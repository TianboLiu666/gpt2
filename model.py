import os
from dataclasses import dataclass
import math
import inspect

import torch
from torch import nn
from torch.nn import functional as F

# batch_size = 12
# n_embd = 768
# block_size = 1024
# max_iters = 5000
# eval_iterval = 500
# eval_iters = 200
# learning_rate = 3e-4
# n_layer = 12
# n_head = 12
# dropout = 0.2

# bias = True

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)
# print(torch.backends.mps.is_available())

# torch.manual_seed(1337)

# with open("input.txt", "r", encoding="utf-8") as f:
#     text = f.read()

# chars = sorted(list(set(text)))
# vocab_size = len(chars)

# stoi = {ch: i for i, ch in enumerate(chars)}
# itos = {i: ch for i, ch in enumerate(chars)}

# encode = lambda s: [stoi[c] for c in s]
# decode = lambda l: "".join([itos[i] for i in l])

# data = torch.tensor(encode(text), dtype=torch.long)
# n = int(0.9 * len(data))
# train_data = data[:n]
# val_data = data[n:]


# def get_batch(split):
#     data = train_data if split == "train" else val_data
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     x = torch.stack([data[i : i + block_size] for i in ix])
#     y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

#     x, y = x.to(device), y.to(device)
#     return x, y


# class LayerNorm(nn.Module):
#     def __init__(self, ndim, bias):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(ndim))
#         self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

#     def forward(self, input):
#         return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.key = nn.Linear(n_embd, n_embd, bias=False)
        # self.query = nn.Linear(n_embd, n_embd, bias=False)
        # self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )

        # self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.shape
        # k = self.key(x)
        # q = self.query(x)
        # v = self.value(x)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        assert self.n_embd % self.n_head == 0

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1,
            2,
        )
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1,
            2,
        )
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1,
            2,
        )

        # compute attention scores ('affinities')
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)

        out = wei @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))

        return out


# class MultiHeadAttention(nn.Module):
#     def __init__(self, num_heads, head_size) -> None:
#         super().__init__()
#         self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
#         self.proj = nn.Linear(n_embd, n_embd)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         out = torch.cat([h(x) for h in self.heads], dim=-1)
#         out = self.dropout(self.proj(out))
#         return out


class FeedFoward(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(n_embd, 4 * n_embd),
        #     nn.ReLU(),
        #     nn.Linear(4 * n_embd, n_embd),
        #     nn.Dropout(dropout),
        # )
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # head_size = n_embd // n_head
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = FeedFoward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight
        # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self):
        return (
            sum(p.numel() for p in self.parameters())
            - self.transformer.wpe.weight.numel()
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.shape
        assert T <= self.config.block_size, "input too long"
        token_emb = self.transformer.wte(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)
        # token_emb = self.transformer.wte(idx)
        # pos_emb = self.transformer.wpe(torch.arange(T, dtype=torch.long, device=device))
        x = self.transformer.drop(token_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
            logits = logits[:, [-1], :]
        else:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config = dict(n_layer=12, n_head=12, n_embd=768)
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config["bias"] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        # if "dropout" in override_args:
        #     print(f"overriding dropout rate to {override_args['dropout']}")
        config["dropout"] = 0.1
        # create a from-scratch initialized minGPT model

        gptConfig = GPTConfig(**config)

        model = GPT(gptConfig)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.tril")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        # sd_keys_hf = [
        #     k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        # ]  # ignore these, just a buffer
        # sd_keys_hf = [
        #     k for k in sd_keys_hf if not k.endswith(".attn.bias")
        # ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)

            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# model = GPT()

# m = model.to(device)

# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# for iter in range(max_iters):
#     if iter % eval_iterval == 0:
#         losses = estimate_loss()
#         print(
#             f'step {iter}: train loss {losses["train"]:.4f} val loss {losses["val"]:.4f}'
#         )

#     xb, yb = get_batch("train")

#     logits, loss = model(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

if __name__ == "__main__":
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        vocab_size=None,
        dropout=dropout,
    )
    gpt = GPT.from_pretrained("gpt2")
