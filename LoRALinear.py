import math

import torch
import torch.nn as nn

class LoRALinear(nn.Module):

    def __init__(
        self,
        base_linear,
        rank=8,
        alpha=16,
        dropout=0.0,
    ):

        super().__init__()

        self.base_linear = (
            base_linear
        )

        self.in_features = (
            base_linear.in_features
        )

        self.out_features = (
            base_linear.out_features
        )

        self.rank = rank
        self.alpha = alpha

        self.scaling = (
            alpha / rank
        )


        # Freeze Linear gốc
        for param in (
            self.base_linear.parameters()
        ):

            param.requires_grad = False


        # A:
        # in_features -> rank

        self.lora_A = nn.Linear(
            self.in_features,
            rank,
            bias=False,
        )


        # B:
        # rank -> out_features

        self.lora_B = nn.Linear(
            rank,
            self.out_features,
            bias=False,
        )


        self.dropout = nn.Dropout(
            dropout
        )


        # A random
        nn.init.kaiming_uniform_(
            self.lora_A.weight,
            a=math.sqrt(5),
        )

        # B = 0
        nn.init.zeros_(
            self.lora_B.weight
        )


    def forward(self, x):

        base_output = (
            self.base_linear(x)
        )


        lora_output = (
            self.dropout(x)
        )

        lora_output = (
            self.lora_A(
                lora_output
            )
        )

        lora_output = (
            self.lora_B(
                lora_output
            )
        )


        return (
            base_output
            +
            self.scaling
            *
            lora_output
        )


def freeze_model(model):

    for param in (
        model.parameters()
    ):

        param.requires_grad = False


def apply_lora(
    module,
    target_names=(
        "query",
        "value",
    ),
    rank=8,
    alpha=16,
    dropout=0.0,
):

    for name, child in list(
        module.named_children()
    ):

        if (
            name in target_names
            and
            isinstance(
                child,
                nn.Linear,
            )
        ):

            setattr(
                module,
                name,
                LoRALinear(
                    base_linear=child,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                ),
            )

        else:

            apply_lora(
                child,
                target_names=target_names,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )

def count_parameters(model):

    total = sum(
        param.numel()

        for param
        in model.parameters()
    )


    trainable = sum(
        param.numel()

        for param
        in model.parameters()

        if param.requires_grad
    )


    percentage = (
        100
        *
        trainable
        /
        total
    )


    print(
        f"Total parameters: "
        f"{total:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable:,}"
    )

    print(
        f"Trainable percentage: "
        f"{percentage:.4f}%"
    )



def get_trainable_parameters(
    model,
):

    return [
        param

        for param
        in model.parameters()

        if param.requires_grad
    ]


# ============================================================
# SAVE ONLY LoRA
# ============================================================

def get_lora_state_dict(
    model,
):

    state = {}

    for name, value in (
        model.state_dict().items()
    ):

        if (
            "lora_A" in name
            or
            "lora_B" in name
        ):

            state[name] = (
                value.cpu()
            )


    return state