# GPT2
## Model Architecture
### Vanilla Model Architecture
MultiHead Attention
* input x (batch_size, seq_length, d_model)
* project x to q, k, v 
* split heads
* attention score = Softmax((q @ k^T) / sqrt(d_model // num_heads)) @ V (batch_size, num_heads, seq_length, seq_length)
* concate back to (batch_size, seq_length, d_model)

Decorder-only Transformer Block

GPTBlock


### Improved Model Architecture
RoPE

RoPE MultiHead Attention
* input x (batch_size, seq_length, d_model)
* project x to q, k, v 
* split heads
* **apply RoPE to q and k**
* attention score = Softmax((**rope(q) @ rope(k)**^T) / sqrt(d_model // num_heads)) @ V (batch_size, num_heads, seq_length, seq_length)
* concate back to (batch_size, seq_length, d_model)


## Optimization direction
### Scaling efficiency tricks
GQA
Sliding Window Attention
SDPA
MoE (large massive LLM, when the parameter size is about trillon)

### Frontier research ideas
Multi-head Latent Attention



#### Questuons
1. In the model architecture, why use the RoPE, why masking is important to GPT (decoder-only Transformer) and why use torch.triu and register_buffer to initialize the mask?
2. Regarding the attention mechanism optimization (KV cache and Attention efficiency), what is GQA and Flash Attention for? What is the difference among different version of Flash Attention? To more frontier attention, what is MLA, linear attention as DeltaNet, and sliding window attention?
3. What to test for the model, 

python -m models.gpt2.