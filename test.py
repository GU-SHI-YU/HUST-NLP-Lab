from transformers import AlbertModel, BertTokenizer
import torch

  
tokenizer = BertTokenizer.from_pretrained("clue/albert_chinese_tiny")
model = AlbertModel.from_pretrained("clue/albert_chinese_tiny")

tests = []
input_text = "今天"
input_text1  = "你好"
tokenized_text=tokenizer.encode(input_text)
tokenized_text1=tokenizer.encode(input_text1)
tests.append(tokenized_text)
tests.append(tokenized_text1)
print(tokenized_text, len(tokenized_text))
input_ids=torch.tensor(tests).view(-1, 2, len(tokenized_text))
print(input_ids)
outputs=model(input_ids)
print(outputs[0].shape, outputs[1].shape)