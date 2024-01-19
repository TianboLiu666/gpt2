import os
import torch
import tiktoken
from model import GPTConfig, GPT

out_dir = 'out'
start = '\n'
num_samples = 3
max_new_tokens = 500
temperature = 0.8
top_k = 200
seed = 2023
device = 'cuda'
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32"

ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = torch.amp.autocast(device_type=device, dtype=ptdtype)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
# gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(checkpoint['model_args'])
state_dict = checkpoint['model']
model.load_state_dict(state_dict)


model.eval()
model.to(device)

enc = tiktoken.get_encoding('gpt2')
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')