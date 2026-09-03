import torch 
import torch.nn as nn
from torch.nn import functional as F
from BPEtokenizer import BPETokenizer


#========================================================
# Hyperparameters========================================
#========================================================



batch_size = 32
block_size = 64

n_emb = 128
n_head = 4
n_layer = 4

learning_rate = 3e-4

max_steps = 5000


eval_interval = 500

eval_iters = 100

device = "cuda" if torch.cuda.is_available() else "cpu"

#=========================================================
# Read raw-text===========================================
#=========================================================







#===================================================================
#Split Data ========================================================
#===================================================================


#===================================================================
#Get batch =========================================================
#===================================================================

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
class Single_head_attention(nn.Module):
    def __init__(self,n_emb_C, head_size, block_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(n_emb_C,head_size,bias = False)
        self.key = nn.Linear(n_emb_C,head_size,bias = False)
        self.value = nn.Linear(n_emb_C,head_size,bias = False)

        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
    def forward(self,x):
        B,T,H = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        # can use T insteal self.head_size
        attention_weight = (q @ (k.transpose(-2,-1)))/self.head_size ** 0.5

        attention_weight = attention_weight.masked_fill(self.tril[:T,:T] == 0, float('-inf'))

        attention_weight = F.softmax(attention_weight, dim = -1)

        output = attention_weight @ v

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self,n_emb_C,n_head_H,block_size):
        super().__init__()
        assert( n_emb_C % n_head_H == 0 )
        head_size = n_emb_C // n_head_H

        self.heads = nn.ModuleList([Single_head_attention(n_emb_C,head_size,block_size) for _ in range(n_head_H)])
        #use this after concat
        self.projection = nn.Linear(n_emb_C,n_emb_C)\

    def forward(self,x):

        out = torch.cat([head(x) for head in self.heads], dim= -1
        
    ) 

        return self.projection(out)

class FeedFowardNetwork(nn.Module):
    def __init__(self, n_emb_C):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_emb_C,n_emb_C*4),
            nn.GELU(),
            nn.Linear(n_emb_C*4,n_emb_C)
        )
    def forward(self,x):
        return self.net(x)


class Blocks(nn.Module):
    def __init__(self,n_emb_C,n_head_H,block_size):
        super().__init__()
        self.self_attention = MultiHeadAttention(n_emb_C,n_head_H,block_size)

        self.feed_forward = FeedFowardNetwork(n_emb_C)

        self.layer_norm_1 = nn.LayerNorm(n_emb_C)

        self.layer_norm_2 = nn.LayerNorm(n_emb_C)
    def forward(self,x):

        x = x + self.self_attention(self.layer_norm_1(x))

        x = x + self.feed_forward(self.layer_norm_2(x))

        return x

    
    
class MiniChatGPT(nn.Module):
    def __init__(self,  
        vocal_size_v,
        block_size,
        n_emb_C,
        n_head_H,
        n_layer
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocal_size_v,n_emb_C)

        self.position_embedding = nn.Embedding(block_size,n_emb_C)

        self.blocks = nn.Sequential(*[Blocks(n_emb_C,n_head_H,block_size) for _ in range(n_layer)])

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
    def generate(self, idx, max_new_token):

        for _ in range(max_new_token):
            idx_cond = idx[:,-self.block_size:]

            logits,_ = self(idx_cond)

            logit = logits[:,-1,:]

            probs = F.softmax(logit,dim=-1)

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
    step
):
    checkpoint = {
        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "step": step,

        "config": {
            "vocab_size": vocab_size,
            "block_size": block_size,
            "n_emb": n_emb,
            "n_head": n_head,
            "n_layer": n_layer,
        },

        "tokenizer": {
            "num_merges":
                tokenizer.num_merges,

            "merges":
                tokenizer.merges,

            "token_to_id":
                tokenizer.token_to_id,

            "id_to_token":
                tokenizer.id_to_token,
        }
    }

    torch.save(
        checkpoint,
        path
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
if __name__ == "__main__":
    with open("./clawer/input.txt","r",encoding = "utf-8") as f:
        text = f.read()



    #=========================================================
    # Encoding================================================
    #=========================================================



    tokenizer = BPETokenizer(
        num_merges=100
    )
    tokenizer.fit(
        text.splitlines()
    )

    vocab_size = tokenizer.vocab_size

    token_ids = tokenizer.encode(text)

    data = torch.tensor(token_ids, dtype=torch.long)

    n = int(0.9 * len(data))

    train_data = data[:n]
    val_data = data[n:]

    model = MiniChatGPT(vocal_size_v = vocab_size,
        block_size = block_size,
        n_emb_C = n_emb,
        n_head_H = n_head,
        n_layer = n_layer)

    model = model.to(device)
    num_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)
    print(
        f"Model parameters: "
        f"{num_parameters:,}"
    )

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
        if step % eval_interval == 0:
            losses = estimate_loss()
            print(
                f"step {step:5d} | "
                f"train loss "
                f"{losses['train']:.4f} | "
                f"val loss "
                f"{losses['val']:.4f}"
            )

        xb,yb = get_batch('train')

        logits,loss,_ = model(xb,yb)

        optimizer.zero_grad()
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_( model.parameters(), max_norm=1.0)

        optimizer.step()
