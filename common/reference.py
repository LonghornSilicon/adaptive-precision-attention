import torch
import math


def reference_attention(Q, K, V, causal=False, block_size=128):
    """Tiled FP64 reference attention using online softmax.

    Computes exact attention without materializing the full N×N score matrix.
    Uses the same online softmax algorithm as FlashAttention.

    Args:
        Q: (B, H, N, d) float64
        K: (B, H, N, d) float64
        V: (B, H, N, d) float64
        causal: apply causal mask
        block_size: tile size for tiled computation

    Returns:
        O: (B, H, N, d) float64
    """
    B, H, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    Br = block_size
    Bc = block_size
    Tr = math.ceil(N / Br)
    Tc = math.ceil(N / Bc)

    O = torch.zeros_like(Q)
    L = torch.zeros(B, H, N, 1, dtype=Q.dtype, device=Q.device)
    M = torch.full((B, H, N, 1), float("-inf"), dtype=Q.dtype, device=Q.device)

    for i in range(Tr):
        q_start = i * Br
        q_end = min(q_start + Br, N)
        Q_block = Q[:, :, q_start:q_end, :]

        o_block = torch.zeros_like(Q_block)
        m_block = torch.full(
            (B, H, q_end - q_start, 1), float("-inf"),
            dtype=Q.dtype, device=Q.device,
        )
        l_block = torch.zeros(
            B, H, q_end - q_start, 1, dtype=Q.dtype, device=Q.device,
        )

        j_end = Tc if not causal else min(i + 1, Tc)
        for j in range(j_end):
            k_start = j * Bc
            k_end = min(k_start + Bc, N)
            K_block = K[:, :, k_start:k_end, :]
            V_block = V[:, :, k_start:k_end, :]

            S_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) * scale

            if causal:
                q_idx = torch.arange(q_start, q_end, device=Q.device).unsqueeze(1)
                k_idx = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)
                causal_mask = q_idx < k_idx
                S_block.masked_fill_(causal_mask, float("-inf"))

            m_block_old = m_block.clone()
            m_new = torch.max(S_block, dim=-1, keepdim=True).values
            m_block = torch.maximum(m_block, m_new)

            exp_scores = torch.exp(S_block - m_block)
            exp_old = torch.exp(m_block_old - m_block)

            l_block = exp_old * l_block + exp_scores.sum(dim=-1, keepdim=True)
            o_block = exp_old * o_block + torch.matmul(exp_scores, V_block)

        o_block = o_block / l_block
        O[:, :, q_start:q_end, :] = o_block
        L[:, :, q_start:q_end, :] = l_block
        M[:, :, q_start:q_end, :] = m_block

    return O
