import argparse
import torch 
import torch.nn as nn
from torch.nn import functional as F
from BPE import BPE
from tokenizers import Tokenizer
from GQA import GroupedQueryAttention
from SwiGLU import SwiGLU
from RMSNorm import RMSNorm

#========================================================
# Hyperparameters========================================
#========================================================



batch_size = 16
block_size = 256

n_emb = 512
n_layer = 12

n_query_heads = 8
n_key_value_heads = 4

learning_rate = 1e-4
max_steps = 10000

eval_interval = 500
eval_iters = 100

device = "cuda" if torch.cuda.is_available() else "cpu"

def apply_p (logits, top_p):
    sorted_logits, sorted_indices = torch.sort(logits,descending=True,dim=-1)
    soft_max = F.softmax(sorted_logits,dim=-1)
    cumulative_probs = torch.cumsum(soft_max,dim=-1)
    sorted_remove = cumulative_probs > top_p
    sorted_remove[:,1:] = sorted_remove[:,:-1].clone() 
    sorted_remove[:, 0] = False
    remove_mask = torch.zeros_like(logits,dtype=torch.bool)
    remove_mask.scatter_(dim=-1,index=sorted_indices,src=sorted_remove)
    logits = logits.masked_fill(remove_mask, float('-inf'))
    return logits
def get_batch(split):
    if split == "train":
        data_source = train_data
    else:
        data_source = val_data
    ix =  torch.randint(0, len(data_source)-block_size,(batch_size,))

    x = torch.stack([data_source[i: i + block_size ] for i in ix])

    y = torch.stack([data_source[i+1:i + block_size +1] for i in ix])
    
    x = x.to(device)
    y = y.to(device)

    return x,y

class FeedFowardNetwork(nn.Module):
    def __init__(self, n_emb_C, hidden_dim = None):
        super().__init__()

        self.net = SwiGLU(n_emb_C, hidden_dim)
    def forward(self,x):
        return self.net(x)


class Blocks(nn.Module):
    def __init__(self,n_emb_C,hidden_dim,n_query_heads,n_key_value_heads,block_size):
        super().__init__()
        self.self_attention = GroupedQueryAttention(n_emb_C,n_query_heads,n_key_value_heads,block_size)

        self.feed_forward = FeedFowardNetwork(n_emb_C,hidden_dim)

        self.RMSNorm_1 = RMSNorm(n_emb_C)

        self.RMSNorm_2 = RMSNorm(n_emb_C)
    def forward(self,x):

        x = x + self.self_attention(self.RMSNorm_1(x))

        x = x + self.feed_forward(self.RMSNorm_2(x))

        return x
class MiniChatGPT(nn.Module):
    def __init__(self,  
        vocal_size_v,
        block_size,
        hidden_dim,
        n_emb_C,
        n_query_heads,
        n_key_value_heads,
        n_layer
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocal_size_v,n_emb_C)

        self.position_embedding = nn.Embedding(block_size,n_emb_C)

        self.blocks = nn.Sequential(*[Blocks(n_emb_C,hidden_dim,n_query_heads,n_key_value_heads,block_size) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_emb_C)

        self.lm_head = nn.Linear(n_emb_C,vocal_size_v)
    def forward(self,idx,targets = None):

        B,T = idx.shape
        # token embedding
        tok_emb = self.token_embedding(idx)
        # get position
        position = torch.arange(T,device=idx.device)
        # position embedding
        pos_emb = self.position_embedding(position)

        x = tok_emb + pos_emb

        x = self.blocks(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            B,T,V = logits.shape

            logits_flatten = logits.reshape(B*T,V)

            targets_flatten = targets.reshape(B*T)


            loss = F.cross_entropy(logits_flatten,targets_flatten)

        return logits,loss
    @torch.no_grad()
    def generate(self, idx, max_new_token,top_p = None,top_k = None,temperature = 1,do_sample = True):


        for _ in range(max_new_token):
            idx_cond = idx[:,-self.block_size:]

            logits,_ = self(idx_cond)

            logits = logits[:,-1,:]
            if not do_sample:
                idx_next = torch.argmax(logits,dim = -1,keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    k = min(top_k, logits.size(-1))   
                    top_values,_ = torch.topk(logits,k=k,dim=-1)
                    thread_hold = (top_values[:,-1].unsqueeze(-1))
                    logits = logits.masked_fill(logits < thread_hold, float('-inf'))
                if top_p and 0 < top_p < 1.0:
                    logits = apply_p(logits,top_p)
                probs = F.softmax(logits,dim=-1)
                idx_next = torch.multinomial(probs,num_samples=1)

            idx = torch.cat((idx,idx_next), dim = 1)

        return idx



# ============================================================
#  OPTIMIZER =================================================
# ============================================================


def save_checkpoint(
    path,
    model,
    optimizer,
    tokenizer,
    step,
):
    checkpoint = {
        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "step": step,

        "config": {
            "vocab_size":
                tokenizer.get_vocab_size()
                if hasattr(tokenizer, "get_vocab_size")
                else tokenizer.vocab_size,

            "block_size":
                block_size,

            "n_emb":
                n_emb,

            "n_query_heads":
                n_query_heads,

            "n_key_value_heads":
                n_key_value_heads,

            "n_layer":
                n_layer,
        },
    }

    torch.save(
        checkpoint,
        path,
    )
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}

    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb,yb = get_batch(split)
            _,loss = model(xb,yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()      
    model.train()     
    return out


def load_tokenizer(name, path):
    """Load one of the supported tokenizers and expose a common interface."""
    if name == "hf":
        tokenizer = Tokenizer.from_file(path)
        vocab_size = tokenizer.get_vocab_size()
        encode = lambda text: tokenizer.encode(text).ids
        decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)
    elif name == "custom-bpe":
        tokenizer = BPE.load(path)
        vocab_size = tokenizer.vocab_size
        encode = tokenizer.encode
        decode = tokenizer.decode
    else:
        raise ValueError(f"Unknown tokenizer: {name}")
    return tokenizer, vocab_size, encode, decode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MiniGPT")
    parser.add_argument(
        "--tokenizer", choices=["hf", "custom-bpe"], default="hf",
        help="Tokenizer backend (default: hf)",
    )   
    parser.add_argument(
        "--tokenizer-path", default="./tokenizer/tokenizer_hf.json",
        help="Path to the tokenizer JSON/model file",
    )
    parser.add_argument(
        "--input", default="./clawer/input_hf.txt",
        help="Training text file",
    )
    args = parser.parse_args()

    print(f"[MiniGPT] loading tokenizer: {args.tokenizer} -> {args.tokenizer_path}", flush=True)
    tokenizer, vocab_size, encode_text, decode_text = load_tokenizer(
        args.tokenizer, args.tokenizer_path
    )
    print(f"[MiniGPT] tokenizer loaded, vocab_size={vocab_size:,}", flush=True)

    # Build the model before tokenizing the corpus so startup reports its
    # parameter count immediately, even when custom BPE encoding is slow.
    model = MiniChatGPT(
        vocal_size_v=vocab_size,
        block_size=block_size,
        n_emb_C=n_emb,
        n_query_heads=n_query_heads,
        n_key_value_heads=n_key_value_heads,
        n_layer=n_layer,
        hidden_dim=None,
    ).to(device)
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"[MiniGPT] device={device} | parameters={num_parameters:,}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"[MiniGPT] reading input: {args.input}", flush=True)
    with open(args.input, "r", encoding="utf-8") as f:
        texts = f.read()
    print(f"[MiniGPT] input loaded: {len(texts):,} characters", flush=True)



    #=========================================================
    # Encoding================================================
    #=========================================================

    print("[MiniGPT] encoding corpus...", flush=True)
    token_ids = encode_text(texts)
    print(f"[MiniGPT] encoding done: {len(token_ids):,} tokens", flush=True)

    data = torch.tensor(token_ids, dtype=torch.long)

    n = int(0.9 * len(data))

    train_data = data[:n]
    val_data = data[n:]

    for step in range(max_steps):
        model.train()
        if step % 1000 == 0:
            save_checkpoint(
                "minigpt_checkpoint.pt",
                model,
                optimizer,
                tokenizer,
                step
            )
        if step % eval_iters == 0:
            losses = estimate_loss()
            print(
                f"step {step:5d} | "
                f"train loss "
                f"{losses['train']:.4f} | "
                f"val loss "
                f"{losses['val']:.4f}"
            )

        xb,yb = get_batch('train')

        logits,loss = model(xb,yb)

        optimizer.zero_grad()
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_( model.parameters(), max_norm=1.0)

        optimizer.step()
