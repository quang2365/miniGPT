from BPE import BPE
import os
def data(path):
    with open(path,'r',encoding='utf-8') as f:
        d = f.read()
        return d
def train(data, num_merge = 2000) -> BPE:
    tokenizer = BPE(num_merge)
    tokenizer.train(data)
    return tokenizer
def main():


    path = './clawer/input_hf.txt'
    if not os.path.isfile(path):
            print(f"[Tokenizer] file not found: {path}")
            return
    num_merge = 8000

    print(f"[Tokenizer] loading: {path}", flush=True)

    texts = data(path)
    print(f"[Tokenizer] input chars: {len(texts):,}", flush=True)
    print(f"[Tokenizer] target merges: {num_merge:,}", flush=True)
    tokenizer = train(texts,num_merge)
    print(f"[Tokenizer] actual merges: {len(tokenizer.merge):,}", flush=True)
    print(f"[Tokenizer] vocab size: {tokenizer.vocab_size:,}", flush=True)

    num = 0 
    name = f'./tokenizer/tokenizer_{num}.json'
    parent_dir = os.path.dirname(name)
    if parent_dir:
        os.makedirs(parent_dir,exist_ok=True)
    while True:
        if os.path.exists(name):
            num += 1 
            name = f'./tokenizer/tokenizer_{num}.json'
        break
    tokenizer.save(name)
    if os.path.exists(name):
        print(f"[Tokenizer] saved: {name}", flush=True)
if __name__  == '__main__':
    main()
