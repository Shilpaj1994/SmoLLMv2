# SpeedUp Transformers

- Weight Sharing
- Residual standard scaling issue



## SpeedUp

- mantice of 10 `high` instead of `highest`
- use `bfloat16` only for logits and loss
- torch.compile()
- attention mechanism modification `Flash Attention`
- vocab size as a multiple of 2^ power



---



## Changes

- Checkpoint: 2000
- silu instead of gelu
- block size: 2048
- vocab_size: 49152
- n_layer: 30
- n_head: 9
- 





The key changes we made were:

- Removed position embeddings in favor of RoPE (Rotary Position Embeddings)

- Split the attention into separate Q,K,V projections with reduced dimensions for K,V

- Kept the MLP structure with gate/up/down projections

These changes align your model architecture more closely with modern transformer designs like Llama, while maintaining efficiency through reduced KV dimensions