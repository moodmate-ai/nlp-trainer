# Pytorch Framwork

- infra
- core
    - model
    - trainer
    - infer
    - data
    - 
    - dataset loader


# Tokenizer 
number of vocabulary = 32000,
and titans uses exactly same size of vocs
```
from transformers import LlamaTokenizerFast

tokenizer = LlamaTokenizerFast.from_pretrained(
    "hf-internal-testing/llama-tokenizer",
)

len(tokenizer)
```