import torch
import math
import time
import torch.nn as nn
import torch.nn.functional as F

class Decode(nn.Module):
    def __init__(self, h_qk=4, h_v=8, d_head=128):
        super().__init__()
        self.h_qk = h_qk
        self.h_v = h_v
        self.d_head = d_head
        self.scale = 1.0 / math.sqrt(d_head)

    def matmul(a: torch.Tensor, b: torch.Tensor):
        """Float32 matmul for numerical stability."""
        return a.float() @ b.float()

    def run(self, q, k, v, state, A_log, a_param, dt_bias, b_param, scale=None):
        """
        Gated Delta Net decode reference implementation (k-last layout).
        
        State layout: [B, H, V, K] (k-last, K dimension at the end)
        
        Gate computation:
        g = exp(-exp(A_log) * softplus(a + dt_bias))
        beta = sigmoid(b)
        
        Delta rule update:
        state_new = g * state_old + k^T @ (beta * v + (1-beta) * k @ state_old) - k^T @ (k @ state_old)
        output = scale * q @ state_new
        """
        B, T, num_q_heads, K = q.shape
        num_v_heads = v.shape[2]
        device = q.device
        
        if scale is None:
            scale = self.scale
        
        # Compute g and beta from raw parameters
        # TODO: field into the later for loop (only need to compute its own part)
        x = a_param.float() + dt_bias.float()  # [B, 1, HV]
        g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))  # [B, 1, HV]
        beta = torch.sigmoid(b_param.float())  # [B, 1, HV]
        
        # Prepare tensors for the recurrent update loop
        # 1. Dimension Reduction: since it is single token, dimension at position 1 would be 1, so 
        # changing the shape from [B, 1, H, D] to [B, H, D] makes it easier to index into specific heads during the for loops later in the script.
        # 2. Type Conversion: converts the tensors from their original data type for stability
        q_f32 = q.squeeze(1).float()
        k_f32 = k.squeeze(1).float()
        v_f32 = v.squeeze(1).float()
        g_f32 = g.squeeze(1).float()
        beta_f32 = beta.squeeze(1).float()
        
        # If no prior state exists, it initializes a zeroed matrix S
        if state is None:
            state_f32 = torch.zeros(B, num_v_heads, v.shape[3], K, dtype=torch.float32, device=device)
        else:
            state_f32 = state.float()
        
        # Head Alignment: transforms Q and K from 4 heads to 8 heads
        # Ensure that every one of the 8 V heads has a corresponding Q and K head to pair with during the Delta Rule update.
        q_exp = q_f32.repeat_interleave(num_v_heads // num_q_heads, dim=1)
        k_exp = k_f32.repeat_interleave(num_v_heads // num_q_heads, dim=1)
        
        # Memory Pre-allocation: 
        # Create a new tensor of zeros with the exact same shape and data type as your input state matrix
        # Create a tensor to hold the final result of the attention calculation for the current token.
        new_state = torch.zeros_like(state_f32)
        output = torch.zeros(B, num_v_heads, v.shape[3], dtype=torch.float32, device=device)
        
        # B: batch size
        for b_idx in range(B):
            for h_idx in range(self.h_v):
                # Extract the data for a single attention head from the larger tensors
                q_h = q_exp[b_idx, h_idx]
                k_h = k_exp[b_idx, h_idx]
                v_h = v_f32[b_idx, h_idx]
                # Retrieve the recurrent memory state S for the current head
                h_state = state_f32[b_idx, h_idx].clone().transpose(-1, -2)  # [V,K] -> [K,V]
                # Parameter Extraction
                g_val = g_f32[b_idx, h_idx]
                beta_val = beta_f32[b_idx, h_idx]
                
                # Delta rule update
                old_state = g_val * h_state
                old_v = k_h @ old_state
                new_v = beta_val * v_h + (1 - beta_val) * old_v
                state_remove = k_h.unsqueeze(1) @ old_v.unsqueeze(0)
                state_update = k_h.unsqueeze(1) @ new_v.unsqueeze(0)
                # new state
                h_state = old_state - state_remove + state_update
                
                # Get inference output: Query @ state matrix
                output[b_idx, h_idx] = scale * (q_h @ h_state)
                new_state[b_idx, h_idx] = h_state.transpose(-1, -2)  # [K,V] -> [V,K]
        
        output = output.unsqueeze(1).to(torch.bfloat16)
        return output, new_state


# --- Usage with GDN Hyperparameters ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # Parameters
    B = 8
    T = 1
    H_V = 8
    H_QK = 4
    K = 128
    V = 128

    decode_layer = Decode(h_qk=H_QK, h_v=H_V, d_head=K)

    # Inputs
    q = torch.randn(B, T, H_QK, K, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, T, H_QK, K, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, T, H_V, V, device=device, dtype=torch.bfloat16)
    A_log = torch.randn(H_V, device=device)
    a_param = torch.randn(B, T, H_V, device=device)
    dt_bias = torch.randn(B, T, H_V, device=device)
    b_param = torch.randn(B, T, H_V, device=device)

    # Warmup
    out, current_state = decode_layer.run(q, k, v, None, A_log, a_param, dt_bias, b_param)

    # Timed Loop
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    iterations = 10
    for _ in range(iterations):
        # pass current_state back in to simulate a real sequence
        out, current_state = decode_layer.run(q, k, v, current_state, A_log, a_param, dt_bias, b_param)
    
    end.record()
    torch.cuda.synchronize()
    
    avg_time = start.elapsed_time(end) / iterations
    print(f"\nResults for B={B}, H_total={H_QK*2+H_V}, D={(H_QK*2+H_V)*K}:")
    print(f"Average Latency per Token: {avg_time:.2f} ms")