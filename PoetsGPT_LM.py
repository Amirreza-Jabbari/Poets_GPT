#!/usr/bin/env python3

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -------------------------------
# Argument parsing and config
# -------------------------------
def get_args():
    p = argparse.ArgumentParser(description='Char-level GPT training & generation')
    p.add_argument('--input_file',   type=str,   default='input.txt')
    p.add_argument('--batch_size',   type=int,   default=128)
    p.add_argument('--block_size',   type=int,   default=256)
    p.add_argument('--max_iters',    type=int,   default=50000)
    p.add_argument('--eval_interval',type=int,   default=500)
    p.add_argument('--eval_iters',   type=int,   default=200)
    p.add_argument('--learning_rate',type=float, default=3e-4)
    p.add_argument('--n_embd',       type=int,   default=384)
    p.add_argument('--n_head',       type=int,   default=6)
    p.add_argument('--n_layer',      type=int,   default=6)
    p.add_argument('--dropout',      type=float, default=0.2)
    p.add_argument('--device',       type=str,   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--checkpoint',   type=str,   default='PoetsGPT.pth')
    p.add_argument('--generate',     action='store_true')
    p.add_argument('--generate_len', type=int,   default=500)
    p.add_argument('--seed',         type=int,   default=1337)
    p.add_argument('--num_workers',  type=int,   default=4)
    return p.parse_args()

args = get_args()
torch.manual_seed(args.seed)

# -------------------------------
# Prepare data
# -------------------------------
text = open(args.input_file, 'r', encoding='utf-8').read()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: ''.join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
split = int(0.9 * len(data))
train_data = data[:split]
val_data   = data[split:]

class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.bs   = block_size
    def __len__(self):
        return len(self.data) - self.bs
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.bs]
        y = self.data[idx+1:idx+self.bs+1]
        return x, y

train_loader = DataLoader(
    CharDataset(train_data, args.block_size),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True
)
val_loader = DataLoader(
    CharDataset(val_data, args.block_size),
    batch_size=max(1, args.batch_size//2),
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True
)

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = {}
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        tmp = []
        for i, (x,y) in enumerate(loader):
            if i>=args.eval_iters: break
            x,y = x.to(device), y.to(device)
            _, loss = model(x,y)
            tmp.append(loss.item())
        losses[split] = sum(tmp)/len(tmp)
    model.train()
    return losses

# -------------------------------
# Model definition
# -------------------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        C = args.n_embd
        self.key   = nn.Linear(C, head_size, bias=False)
        self.query = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(args.block_size, args.block_size)))
        self.drop  = nn.Dropout(args.dropout)
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        scores = (q @ k.transpose(-2,-1)) * (C**-0.5)
        scores = scores.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        attn = self.drop(F.softmax(scores, dim=-1))
        v = self.value(x)
        return attn @ v     # (B,T,hs)

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = args.n_embd // args.n_head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(args.n_head)])
        self.proj  = nn.Linear(args.n_embd, args.n_embd)
        self.drop  = nn.Dropout(args.dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.drop(self.proj(out))

class FeedFoward(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C, 4*C),
            nn.ReLU(),
            nn.Linear(4*C, C),
            nn.Dropout(args.dropout),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        C, H = args.n_embd, args.n_head
        self.sa   = MultiHeadAttention()
        self.ffwd = FeedFoward(C)
        self.ln1  = nn.LayerNorm(C)
        self.ln2  = nn.LayerNorm(C)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        C = args.n_embd
        self.token_embedding_table    = nn.Embedding(vocab_size, C)
        self.position_embedding_table = nn.Embedding(args.block_size, C)
        self.blocks = nn.Sequential(*[Block() for _ in range(args.n_layer)])
        self.ln_f   = nn.LayerNorm(C)
        self.lm_head= nn.Linear(C, vocab_size)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)
    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok = self.token_embedding_table(idx)
        pos = self.position_embedding_table(torch.arange(T, device=idx.device))
        x   = tok + pos
        x   = self.blocks(x)
        x   = self.ln_f(x)
        logits = self.lm_head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new):
        for _ in range(max_new):
            idx_cond = idx[:, -args.block_size:]
            logits, _ = self(idx_cond)
            probs     = F.softmax(logits[:, -1, :], dim=-1)
            nxt       = torch.multinomial(probs, num_samples=1)
            idx       = torch.cat([idx, nxt], dim=1)
        return idx

# -------------------------------
# Train / Generate
# -------------------------------
if __name__ == '__main__':
    device = args.device
    model  = GPTLanguageModel().to(device)

    # CUDA speedups
    use_amp = device.startswith('cuda')
    if use_amp:
        torch.backends.cudnn.benchmark = True
        scaler = torch.amp.GradScaler()
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
            except Exception as e:
                print(f"[warning] torch.compile failed: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # load checkpoint if exists
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # generation only?
    if args.generate:
        ctx = torch.zeros((1,1), dtype=torch.long, device=device)
        out = model.generate(ctx, args.generate_len)[0].tolist()
        txt = decode(out)
        print(txt)
        with open('generated_output.txt','w',encoding='utf-8') as f:
            f.write(txt)
        exit()

    # training loop
    pbar = tqdm(total=args.max_iters, desc='Training', ncols=80)
    it = 0
    while it < args.max_iters:
        for x,y in train_loader:
            if it >= args.max_iters: break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast():
                    _, loss = model(x,y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                _, loss = model(x,y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            it += 1
            pbar.update(1)
            if it % args.eval_interval == 0:
                res = estimate_loss(model)
                pbar.write(f"step {it}: train {res['train']:.4f}, val {res['val']:.4f}")
                torch.save(model.state_dict(), args.checkpoint)

    # final generation
    ctx = torch.zeros((1,1), dtype=torch.long, device=device)
    out = model.generate(ctx, args.generate_len)[0].tolist()
    txt = decode(out)
    print(txt)
    with open('generated_output.txt','w',encoding='utf-8') as f:
        f.write(txt)
