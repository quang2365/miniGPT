from BPE import BPE
from minGPT import MiniChatGPT
from LoRALinear import apply_lora
from tokenizers import Tokenizer
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = Tokenizer.from_file("./tokenizer/tokenizer_hf.json")

checkpoint = torch.load('./minigpt_sft.pth',map_location=device)


config = checkpoint['config']
vocab_size = config['vocab_size']

block_size = config['block_size']

n_emb = config['n_emb']

n_query_heads = config['n_query_heads']

n_key_value_heads = config['n_key_value_heads']

n_layer = config['n_layer']

model = MiniChatGPT(vocal_size_v=vocab_size,
                    block_size= block_size,
                    n_emb_C=n_emb,
                    n_query_heads=n_query_heads,
                    n_key_value_heads = n_key_value_heads,
                    n_layer=n_layer,
                    hidden_dim=None)

# SFT checkpoint contains LoRA-wrapped q/v projections.  Recreate the same
# wrappers before loading the checkpoint state dict.
apply_lora(model, target_names=("q_projec", "v_projec"), rank=8, alpha=16, dropout=0.05)

model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
prompt = str(input('nhập prompt: '))

prompt = tokenizer.encode(prompt).ids

prompt = torch.tensor([prompt],dtype=torch.long,device=device)
model.eval()
with torch.no_grad():   
    out = model.generate(prompt,300,1,1,1,True)
out = out[0].tolist()

result = tokenizer.decode(out,skip_special_tokens=True)

print(result)



