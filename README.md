tokens/update = seq_length × per_device_batch_size × grad_accum × #GPUs
For a 0.6B model, something like 50k–100k tokens/update is very reasonable.
For a 8B model, people often go into the 0.5–2M tokens/update range.

LR ≈ LR_ref × (tokens_per_update / tokens_per_update_ref)
32k tokens/update → 7e-5 – 1e-4
65k tokens/update → 1e-4 – 1.5e-4
130k tokens/update → 1.5e-4 – 2e-4

using packing: true. Packing concatenates many short lines and fills each training sequence up to your cutoff_len (e.g., 4096). So the optimizer “sees” roughly 4096 tokens per sequence regardless of your average 40-token lines.
tokens_per_update ≈ 4096 × 1 × 16 ≈ 65k


to use llama.cpp:

./llama-cli \
  -m /root/models/qwen3-8b-q4/Qwen3-8B-Q4_K_M.gguf \
  -i -c 4096 -ngl 999 \
#  --chat-template default
#  ngl is # of layer loaded onto gpu