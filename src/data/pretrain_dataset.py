import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk

from transformers import AutoTokenizer


class GPTDataset(Dataset):
    def __init__(self, docs: list[str], tokenizer, max_length, stride):
        """
        Initialize GPT Dataset.
        
        Args:
            docs: List of raw text documents
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum sequence length
            stride: Step size for sliding window
        """

        super().__init__()

        # Store args
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.input_ids = None
        self.target_ids = None

        # Encode the entire text into integer token ids
        full_text = "\n\n".join(docs)
        list_of_ids = tokenizer.encode(full_text)
        token_ids = torch.tensor(list_of_ids, dtype=torch.long)

        # Slide a window of size 'max_length' over token_ids with step 'stride'
        L = len(token_ids)
        max_start_idx = max(0, L - max_length)
        start_indices = torch.arange(0, max_start_idx, stride)
        self.input_ids = token_ids[start_indices[:, None] + torch.arange(max_length)].to(torch.long)
        self.target_ids = token_ids[start_indices[:, None] + torch.arange(1, max_length + 1)].to(torch.long)

    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]
    


class GPTArrowDataset(Dataset):
    def __init__(self, arrow_dataset_path: str):
        self.dataset = load_from_disk(arrow_dataset_path)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        example = self.dataset[index]
        input_ids = torch.tensor(example['input_ids'], dtype=torch.long)
        labels = torch.tensor(example['labels'], dtype=torch.long)
        return input_ids, labels

def create_dataloader(txt=None, arrow_dataset_path=None, batch_size=16, max_length=256, stride=128,
                     shuffle=True, drop_last=True, num_workers=0):
    if arrow_dataset_path:
         return DataLoader(GPTArrowDataset(arrow_dataset_path=arrow_dataset_path),
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           drop_last=drop_last,
                                           num_workers=num_workers)
    elif txt is not None:
        return DataLoader(GPTDataset(txt,
                                     tokenizer=setup_tokenizer(),
                                     max_length=max_length,
                                     stride=stride),
                          batch_size=batch_size,
                          shuffle=shuffle,
                          drop_last=drop_last,
                          num_workers=num_workers)
    else:
        raise ValueError


def setup_tokenizer():
    special_tokens_dict = {
        "additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
    }

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="gpt2",
                                              pad_token="<|pad|>")
    tokenizer.add_special_tokens(special_tokens_dict)

    return tokenizer


