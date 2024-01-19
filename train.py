import os
import math
import numpy as np
import torch

from model import GPT

out_dir = "out"
dataset = "berkshire"
eval_iters = 40
eval_interval = 20
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# Total 6,366,608 tokens, so total 194 iters ~ 1 epoch
batch_size = 1

block_size = 1024
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.2
bias = False
learning_rate = 3e-5

decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

max_iters = 400  # epochs
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
gradient_accumulation_steps = 32


device = "cuda"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float32"
)
print(device)
print(torch.backends.mps.is_available())

ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = torch.amp.autocast(device_type=device, dtype=ptdtype)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
config = {k: globals()[k] for k in config_keys}

torch.manual_seed(2023)

# data_dir = os.path.join("data", dataset)

train_data = np.memmap("train.bin", dtype=np.uint16, mode="r")
val_data = np.memmap("val.bin", dtype=np.uint16, mode="r")


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    if device == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# def get_lr(it):
#     # 1) linear warmup for warmup_iters steps
#     if it < warmup_iters:
#         return learning_rate * it / warmup_iters
#     # 2) if it > lr_decay_iters, return min learning rate
#     if it > lr_decay_iters:
#         return min_lr
#     # 3) in between, use cosine decay down to min learning rate
#     decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
#     assert 0 <= decay_ratio <= 1
#     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
#     return min_lr + coeff * (learning_rate - min_lr)


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

model = GPT.from_pretrained("gpt2")
model.to(device)

model_args = model.config

optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device
)

X, Y = get_batch("train")

while True:
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
    if losses["val"] < best_val_loss:
        best_val_loss = losses["val"]
        if iter_num > 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "config": config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = (
                loss / gradient_accumulation_steps
            )  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch("train")
        # backward pass, with gradient scaling if training in fp16
        loss.backward()

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    iter_num += 1

    if iter_num > max_iters:
        print("Done")
        break
