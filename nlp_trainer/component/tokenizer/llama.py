from transformers import LlamaTokenizerFast


class LlamaTokenizer:
    def __init__(self, tokenizer_path: str = "hf-internal-testing/llama-tokenizer"):
        self.tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_path)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)


if __name__ == "__main__":
    tokenizer = LlamaTokenizer()
    print(tokenizer.encode("Hello, world!"))
