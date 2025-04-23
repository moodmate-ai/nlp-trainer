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


tion_llama_fast.LlamaTokenizerFast'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565 - if you loaded a llama tokenizer from a GGUF file you can ignore this message.