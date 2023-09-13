from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir='/data/xcao/huggingface_cache', truncation_side="right")
tokenizer.add_special_tokens({"bos_token": "[DEC]"})