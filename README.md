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

# How to structure the project?
- usecases
    - train
    - eval
    - infer


250,000건 마다 파라미터 저장 -> 하루마다