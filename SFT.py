import json
import random

import torch

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from tokenizers import Tokenizer

from minGPT import MiniChatGPT

from LoRALinear import (
    freeze_model,
    apply_lora,
    count_parameters,
    get_trainable_parameters,
    get_lora_state_dict,
)

# ============================================================
# CONFIG
# ============================================================

BASE_MODEL_PATH = "./minigpt_checkpoint.pt"
TOKENIZER_PATH = "./tokenizer/tokenizer_hf.json"
SFT_DATA_PATH = "./clawer/data/sft_generated_1000.jsonl"
OUTPUT_MODEL_PATH = "./minigpt_sft.pth"

batch_size = 16
learning_rate = 1e-4
num_epochs = 10
val_ratio = 0.1
seed = 42

device = ( "cuda" if torch.cuda.is_available() else "cpu" )

random.seed(seed)
torch.manual_seed(seed)


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = Tokenizer.from_file( TOKENIZER_PATH )

PAD_ID = tokenizer.token_to_id( "<|pad|>" )

BOS_ID = tokenizer.token_to_id( "<|bos|>" )

EOS_ID = tokenizer.token_to_id( "<|eos|>" )

SYSTEM_ID = tokenizer.token_to_id( "<|system|>" )

USER_ID = tokenizer.token_to_id( "<|user|>" )

ASSISTANT_ID = tokenizer.token_to_id( "<|assistant|>" )

vocab_size = tokenizer.get_vocab_size()


def encode_text(text):
    return tokenizer.encode(
        text,
        add_special_tokens=False,
    ).ids


# ============================================================
# LOAD BASE MODEL
# ============================================================

checkpoint = torch.load( BASE_MODEL_PATH, map_location=device, )

config = checkpoint["config"]

block_size = config["block_size"]
n_emb = config["n_emb"]
n_query_head = config["n_query_heads"]
n_key_value_heads = config["n_key_value_heads"]
n_layer = config["n_layer"]

model = MiniChatGPT( vocal_size_v=vocab_size, block_size=block_size, n_emb_C=n_emb, n_query_heads= n_query_head,n_key_value_heads=n_key_value_heads, n_layer=n_layer, hidden_dim = None )
model.load_state_dict( checkpoint[ "model_state_dict" ] )

freeze_model(model)

apply_lora(model,target_names=("q_projec","v_projec"),rank=8,alpha=16,dropout=0.05)

model = model.to(device)

count_parameters(model)

# ============================================================
# BUILD SFT SAMPLE
# ============================================================

def build_sft_sample(sample):

    system_text = ( sample.get( "system", "" ).strip() )

    user_text = ( sample.get( "user", "" ).strip() )

    assistant_text = ( sample.get( "assistant", "" ).strip() )
    if (
        not user_text
        or
        not assistant_text
    ):
        return None


    full_ids = []
    loss_mask = []


    # --------------------------------------------------------
    # BOS
    # --------------------------------------------------------

    full_ids.append(
        BOS_ID
    )

    loss_mask.append(
        0
    )


    # --------------------------------------------------------
    # SYSTEM
    # --------------------------------------------------------

    if system_text:

        full_ids.append(SYSTEM_ID)
        loss_mask.append(0)
        system_id = encode_text(system_text)
        full_ids.extend(system_id)
        loss_mask.extend([0] * len(system_id))
    # --------------------------------------------------------
    # USER
    # --------------------------------------------------------
    full_ids.append(USER_ID)
    loss_mask.append(0)
    user_ids = encode_text(user_text)
    full_ids.extend(user_ids)
    loss_mask.extend([0] * len(user_ids))
    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    full_ids.append(ASSISTANT_ID)
    loss_mask.append(0)
    assistant_ids = encode_text(assistant_text)
    full_ids.extend(assistant_ids)
    loss_mask.extend([1] * len(assistant_ids))
    # --------------------------------------------------------
    # EOS
    # --------------------------------------------------------

    full_ids.append(EOS_ID)
    loss_mask.append(1)

    # --------------------------------------------------------
    # LENGTH CHECK
    # --------------------------------------------------------

    if len(full_ids) > (
        block_size + 1
    ):
        return None


    # --------------------------------------------------------
    # NEXT-TOKEN SHIFT
    # --------------------------------------------------------

    input_ids = (
        full_ids[:-1]
    )

    targets = (
        full_ids[1:]
    )

    target_mask = (
        loss_mask[1:]
    )


    labels = [
        token_id
        if mask == 1
        else -100

        for token_id, mask
        in zip(
            targets,
            target_mask,
        )
    ]


    return {
        "input_ids":
            torch.tensor(
                input_ids,
                dtype=torch.long,
            ),

        "labels":
            torch.tensor(
                labels,
                dtype=torch.long,
            ),
    }


# ============================================================
# DATASET
# ============================================================

class SFTDataset(Dataset):

    def __init__(
        self,
        samples,
    ):

        self.data = []

        for sample in samples:

            item = (
                build_sft_sample(
                    sample
                )
            )

            if item is not None:

                self.data.append(
                    item
                )


    def __len__(self):

        return len(
            self.data
        )


    def __getitem__(
        self,
        index,
    ):

        return self.data[
            index
        ]


# ============================================================
# COLLATE
# ============================================================

def collate_sft_batch(batch):

    max_len = max(
        item[
            "input_ids"
        ].size(0)

        for item
        in batch
    )


    batch_inputs = []
    batch_labels = []


    for item in batch:

        input_ids = (
            item[
                "input_ids"
            ]
        )

        labels = (
            item[
                "labels"
            ]
        )


        pad_length = (
            max_len
            -
            input_ids.size(0)
        )


        if pad_length > 0:

            input_pad = torch.full(
                (
                    pad_length,
                ),
                PAD_ID,
                dtype=torch.long,
            )

            label_pad = torch.full(
                (
                    pad_length,
                ),
                -100,
                dtype=torch.long,
            )

            input_ids = torch.cat(
                [
                    input_ids,
                    input_pad,
                ]
            )

            labels = torch.cat(
                [
                    labels,
                    label_pad,
                ]
            )


        batch_inputs.append(
            input_ids
        )

        batch_labels.append(
            labels
        )


    return {
        "input_ids":
            torch.stack(
                batch_inputs
            ),

        "labels":
            torch.stack(
                batch_labels
            ),
    }


# ============================================================
# LOAD DATA
# ============================================================

with open(
    SFT_DATA_PATH,
    "r",
    encoding="utf-8",
) as f:
    # Hỗ trợ cả JSON array và JSONL (mỗi dòng một object).
    if SFT_DATA_PATH.lower().endswith(('.jsonl', '.ndjson')):
        samples = [json.loads(line) for line in f if line.strip()]
    else:
        samples = json.load(f)

if not isinstance(samples, list) or not samples:
    raise ValueError(f"Dataset phải là danh sách mẫu, file rỗng hoặc sai định dạng: {SFT_DATA_PATH}")


random.shuffle(
    samples
)


split_index = int(
    len(samples)
    *
    (1 - val_ratio)
)


train_samples = (
    samples[:split_index]
)

val_samples = (
    samples[split_index:]
)


train_dataset = SFTDataset(
    train_samples
)

val_dataset = SFTDataset(
    val_samples
)


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=(
        collate_sft_batch
    ),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=(
        collate_sft_batch
    ),
)


# ============================================================
# OPTIMIZER
# ============================================================

trainable_parameters = (
    get_trainable_parameters(
        model
    )
)

optimizer = (
    torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
    )
)


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate():

    model.eval()

    total_loss = 0.0
    total_batches = 0


    for batch in val_loader:

        input_ids = (
            batch[
                "input_ids"
            ].to(device)
        )

        labels = (
            batch[
                "labels"
            ].to(device)
        )


        _, loss = model(
            input_ids,
            labels,
        )


        total_loss += (
            loss.item()
        )

        total_batches += 1


    model.train()


    if total_batches == 0:
        return float("nan")


    return (
        total_loss
        /
        total_batches
    )


# ============================================================
# TRAIN
# ============================================================

best_val_loss = float(
    "inf"
)


for epoch in range(
    num_epochs
):

    model.train()

    total_train_loss = 0.0
    total_batches = 0


    for batch in train_loader:

        input_ids = (
            batch[
                "input_ids"
            ].to(device)
        )

        labels = (
            batch[
                "labels"
            ].to(device)
        )


        _, loss = model(
            input_ids,
            labels,
        )


        optimizer.zero_grad()

        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=1.0,
        )


        optimizer.step()


        total_train_loss += (
            loss.item()
        )

        total_batches += 1


    train_loss = (
        total_train_loss
        /
        max(
            total_batches,
            1,
        )
    )


    val_loss = evaluate()


    print(
        f"Epoch "
        f"{epoch + 1:3d}"
        f"/"
        f"{num_epochs} | "
        f"train loss "
        f"{train_loss:.4f} | "
        f"val loss "
        f"{val_loss:.4f}"
    )


    if val_loss < best_val_loss:

        best_val_loss = (
            val_loss
        )


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "epoch":
                    epoch + 1,

                "train_loss":
                    train_loss,

                "val_loss":
                    val_loss,

                "config":
                    config,

                "stage":
                    "sft",
            },

            OUTPUT_MODEL_PATH,
        )
