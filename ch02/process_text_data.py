"""
分词 -> gen token id -> sliding window sampling ->  gen embd
"""
import os 
import re
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}
    
    def encode(self, text):
        preprocessed  = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [s.strip() for s in preprocessed if s.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = [self.int_to_str[i] for i in ids]
        text = ' '.join(text)
        text = re.sub(r'\s([,.?_!"()\'])', r'\1', text)
        return text

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}
    
    def encode(self, text):
        preprocessed  = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [s.strip() for s in preprocessed if s.strip()]
        preprocessed = [item if item in self.str_to_int 
                            else '<|unk|>' 
                            for item in preprocessed ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = [self.int_to_str[i] for i in ids]
        text = ' '.join(text)
        text = re.sub(r'\s([,.?_!"()\'])', r'\1', text)
        return text




class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader
def get_vocab():
    """
    获取词表
    """
    file_path = os.path.join(os.path.dirname(__file__), 'the-verdict.txt')

    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    print(f"the number of characters in the text is {len(raw_text)}")
    print(f"the first 100 characters of the text are {raw_text[:100]}")

    preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
    preprocessed = [s.strip() for s in preprocessed if s.strip()]
    print(f">>>> length of txt: {len(preprocessed)}")
    print(f">>>> first 100 tokens: {preprocessed[:100]}")

    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)

    print(f">>>> vocab size: {vocab_size}")

    vocab = {s:i for i, s in enumerate(all_words)}
    for i, item in enumerate(vocab.items()):
        print(f"{item}")
        if i >= 50:
            break
    
    # 词表中添加一些特殊字符
    all_words.extend(["<|endoftext|>", "<|startoftext|>", "<|unk|>"])
    vocab = {s:i for i, s in enumerate(all_words)}
    return vocab



def main():
    ## tokenizer 
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text3 = "你好，你是谁"
    text = " <|endoftext|> ".join([text1, text2, text3])

    print(text)
    ## 1. 使用 SimpleTokenizerV1
    print("----------Using SimpleTokenizerV1----------")
    voc = get_vocab()
    tokenizer = SimpleTokenizerV2(vocab=voc)
    
    encode = tokenizer.encode(text)
    print(f"Encoding text: {encode}")
    print(f"Decoding text: {tokenizer.decode(encode)}")

    ## 2. 使用 tiktoken 
    print("----------Using tiktoken----------")
    import tiktoken # tiktoken                0.9.0
    tokenizer = tiktoken.get_encoding("gpt2")
    encode = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    print(f"Encoding text: {encode}")
    print(f"Decoding text: {tokenizer.decode(encode)}") 

    ## 使用滑动窗口采样数据
    print("----------Using sliding window sampling----------")
    file_path = os.path.join(os.path.dirname(__file__), 'the-verdict.txt')
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    encoded_text = tokenizer.encode(raw_text, allowed_special={'|endoftext|'})
    print(f"Encoding text: {len(encoded_text)}")

    enc_samples = encoded_text[:50]

    context_size = 4
    x = enc_samples[:context_size]
    y = enc_samples[1:context_size+1]
    print(f"x: {x}")
    print(f"y:     {y}")

    print(f"----------token in  sliding window sampling----------")
    for i in range(1, context_size+1):
        context = enc_samples[:i]
        desired = enc_samples[i]
        print(context, "---->", desired)

    print(f"---------- context sliding window sampling----------")
    for i in range(1, context_size+1):
        context = enc_samples[:i]
        desired = enc_samples[i]
        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
    max_length = 4
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length,
                                      stride=1, shuffle=False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(f"first batch: {first_batch}")

    second_batch = next(data_iter)
    print(f"second batch: {second_batch}")
 
    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)  # (batch_size, seq_len, output_dim)

    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print(pos_embeddings.shape)  # (context_length, output_dim)

    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)  # (batch_size, seq_len, output_dim)


if __name__ == "__main__":
    main()