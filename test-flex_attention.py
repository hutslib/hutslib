"""
Author: thuaj@connect.ust.hk
Date: 2025-01-10 16:31:47
LastEditTime: 2025-01-17 11:19:36
Description: test flex attention
Copyright (c) 2025 by thuaj@connect.ust.hk, All Rights Reserved.
"""

import tqdm
from typing import Tuple, Union

import torch
import torch.nn as nn

try:
    from torch.nn.attention.flex_attention import flex_attention

    flex_attention_available = True
except ImportError:
    print(f"[Warning] flex attention need pytorch 2.5.0+ but your version is {torch.__version__}")
    flex_attention_available = False


class FlexAttentionLayer(torch.nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        bias=False,
        dropout=0.1,
    ):
        super().__init__()

        if not flex_attention_available:
            raise NotImplementedError(
                f"[Error] flex attention need pytorch 2.5.0+ but your version is {torch.__version__}"
            )

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.n_embd = dim
        self.n_head = num_heads
        self.bias = bias
        self.dropout = dropout

        self.flex_attention = torch.compile(flex_attention)

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, self.n_embd * 3, bias=self.bias)
        self.q = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
        self.k = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
        self.v = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
        # regularization
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        # self.flash =  hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        # Question(thuaj): is there a more elegant way?
        # https://github.com/pytorch/pytorch/issues/133254#issuecomment-2408710459
        self.kernel_options = {
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            "BLOCK_M1": 16,
            "BLOCK_N1": 32,
            "BLOCK_M2": 32,
            "BLOCK_N2": 16,
        }

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        rel_pe,
        relation,
    ):
        # print(">> Starting FlexAttentionLayer forward<<")
        if isinstance(x, torch.Tensor):
            B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
            # print(f"self attention: B={B}, T={T}, C={C}")
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        else:
            src, tgt = x
            B, T, C = src.size()
            _, S, _ = tgt.size()
            # print(f"cross attention: B={B}, T={T}, S={S}, C={C}")
            q = self.q(src).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            k = self.k(tgt).view(B, S, self.n_head, C // self.n_head).transpose(1, 2)
            v = self.v(tgt).view(B, S, self.n_head, C // self.n_head).transpose(1, 2)


        def score_mod(score, b, h, q_idx, kv_idx):
            offset = relation[b, q_idx, kv_idx]
            return torch.where(offset > -1, score + rel_pe[b, q_idx, offset, h], float("-inf"))

        # Note(thuaj): Whether to use or not use kernel_options will both lead to failure.
        y = self.flex_attention(query=q, key=k, value=v, score_mod=score_mod, kernel_options=self.kernel_options)
        # y = self.flex_attention(query=q, key=k, value=v, score_mod=score_mod)

        y = y.transpose(1, 2).contiguous().view(B, -1, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y

class CustomModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decoder_depth = 1

        self.cross_atten_layer = nn.ModuleList(
            [FlexAttentionLayer(dim=128, num_heads=8, bias=False, dropout=0.0) for _ in range(self.decoder_depth)]
        )
        self.self_atten_layer = nn.ModuleList(
            [FlexAttentionLayer(dim=128, num_heads=8, bias=False, dropout=0.0) for _ in range(self.decoder_depth)]
        )


    def forwards(self):

        # Problem>>>>
        # If we adjust the bs, T and topk value, sometimes the code will raise an error
        # reduce the bs, T, topk value, somtimes avoid the error
        # example bs=32 T =90 topk=32 will catch an error ->
        # RuntimeError: CUDA error: an illegal memory access was encountered
        # bs=1 T=1 will run ok
        # And the output is more complex using my entire code.
        bs, T, D = 32, 90, 128
        A = torch.randint(8, 12, (1,)).item()
        M = torch.randint(300,400, (1,)).item()
        print(f"bs: {bs}, T: {T}, D: {D}, A: {A}, M: {M}")
        a_token = torch.randn(bs, A, T, D).to(self.device)
        m_token = torch.randn(bs, M, T, D).to(self.device)
        print(f"a_token: {a_token.shape}, m_token: {m_token.shape}")

        topk=3
        n_heads=8
        a_pe = torch.rand(
            bs * T, A, topk, n_heads,
            requires_grad=True, device=self.device
        )

        rand = torch.rand(bs*T, A, A, device=device)
        # Top-k indices along the M dimension
        topk_vals, topk_indices = torch.topk(rand, k=min(topk, A), dim=-1)
        pair_valid_mask = torch.zeros(bs*T, A, A, dtype=torch.bool, device=device)
        pair_valid_mask.scatter_(-1, topk_indices, True).to(self.device)

        offsets = torch.arange(min(topk, A), device=self.device).unsqueeze(0).expand(bs * T * A, -1)
        a_relation = torch.full((bs * T, A, A), -1, dtype=torch.long, device=self.device)
        a_relation[pair_valid_mask] = offsets.reshape(-1)


        topk=32
        n_heads=8
        # create learnable rpe for each head. Like T5
        a2m_pe = torch.rand(
            bs * T, A, topk, n_heads,
            requires_grad=True, device=self.device
        )

        rand = torch.rand(bs*T, A, M, device=device)  # Random values for ranking
        topk_vals, topk_indices = torch.topk(rand, k=min(topk, M), dim=-1)
        pair_valid_mask = torch.zeros(bs*T, A, M, dtype=torch.bool, device=device)
        pair_valid_mask.scatter_(-1, topk_indices, True).to(self.device)

        offsets = torch.arange(min(topk, M), device=self.device).unsqueeze(0).expand(bs * T * A, -1)
        a2m_relation = torch.full((bs * T, A, M), -1, dtype=torch.long, device=self.device)
        a2m_relation[pair_valid_mask] = offsets.reshape(-1)

        for i in range(self.decoder_depth):
            print(f"a_pe: {a_pe.shape}, a_relation: {a_relation.shape}")
            print(f"a2m_pe: {a2m_pe.shape}, a2m_relation: {a2m_relation.shape}")
            all_equal_to_minus_one = torch.all(a_relation == -1)
            print(f"a_relation all_equal_to_minus_one: {all_equal_to_minus_one}")
            all_equal_to_minus_one = torch.all(a2m_relation == -1)
            print(f"a2m_relation all_equal_to_minus_one: {all_equal_to_minus_one}")
            a_token = self.cross_atten_layer[i](
                (a_token.reshape(bs * T, A, -1), m_token.reshape(bs * T, M, -1)),
                a2m_pe,
                a2m_relation,
            )
            a_token = self.self_atten_layer[i](
                a_token.reshape(bs * T, A, -1),
                a_pe,
                a_relation,
            )
        return a_token

if __name__ == "__main__":
    # I provided a simplicifed version of the code to mimic the workflow
    # The code catch an error, however, I'm not sure if it is the same error as I faced when I run the entire code
    # Test device 1：
    # nvidia-smi: GeForce RTX 3090 24G NVIDIA-SMI 550.127.05 Driver Version: 550.127.05 CUDA Version: 12.4
    # nvcc --version:
    # nvcc: NVIDIA (R) Cuda compiler driver
    # Copyright (c) 2005-2022 NVIDIA Corporation
    # Built on Wed_Sep_21_10:33:58_PDT_2022
    # Cuda compilation tools, release 11.8, V11.8.89
    # Build cuda_11.8.r11.8/compiler.31833905_0
    # pytorch 2.7.0.dev20250112+cu118
    # Test device 2:
    # nvidia-smi: NVIDIA RTX A6000 48G NVIDIA-SMI 535.98 Driver Version: 535.98 CUDA Version: 12.2
    # nvcc --version:
    # nvcc: NVIDIA (R) Cuda compiler driver
    # Copyright (c) 2005-2022 NVIDIA Corporation
    # Built on Wed_Sep_21_10:33:58_PDT_2022
    # Cuda compilation tools, release 11.8, V11.8.89
    # Build cuda_11.8.r11.8/compiler.31833905_0
    # pytorch 2.7.0.dev20250116+cu118

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epoch = 20
    data_len = 100
    model.train()
    for i in range(epoch):
        for data in tqdm.tqdm(range(data_len)):
            out = model.forwards()
            loss_fn = nn.MSELoss()
            loss = loss_fn(out, torch.randn(out.shape).to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
