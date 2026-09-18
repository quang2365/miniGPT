from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from pathlib import Path
SPECIAL_TOKENS = [
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]

input_file = Path("./clawer/input_hf.txt")
output_file = Path("./tokenizer/tokenizer_hf.json")

tokenizer = Tokenizer(BPE(unk_token="[<|unk|>]"))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()
num_merge = 2000
num_special_token = len(SPECIAL_TOKENS)
num_base_token = 256
vocab_size = num_merge + num_special_token + num_base_token
trainer = BpeTrainer(
    vocab_size= vocab_size,
    min_frequency=2,
    initial_alphabet=ByteLevel.alphabet(),
    special_tokens = SPECIAL_TOKENS,
    show_progress=True,
)

tokenizer.train([str(input_file)], trainer)

output_file.parent.mkdir(parents=True, exist_ok=True)
tokenizer.save(str(output_file))

print("Tokenizer saved:", output_file)
print("Vocab size:", tokenizer.get_vocab_size())