# GPT-2 124M Pre-training

This repo includes the code for implementing the GPT2 124M parameter model from scratch and pre-training it on a tiny Shakespeare dataset using torch's distributed data parallel.

### Future work:
- Add evaluation script.
- Train the model on a large dataset like Fineweb.
- Evaluate model on public datasets
- Do Grouped Query Attention
- Add Rotary Position Embeddings
- Use scaling laws for the number of heads from the OpenELM paper.

#### Acknowledgement:
Thanks to Andrej Karpathy's lecture video on YouTube for explaining everything in detail.
