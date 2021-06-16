import torch
import pickle
from transformers import AlbertModel, BertTokenizer

if __name__ == '__main__':
    model = torch.load('save/model_epoch9.pkl', map_location=torch.device('cpu'))
    output = open('cws_result.txt', 'w', encoding='utf-8')
    label_output = open('./data/cws_res.txt', 'w', encoding='utf8')
    tokenizer = BertTokenizer.from_pretrained("clue/albert_chinese_tiny")

    with open('data/datasave.pkl', 'rb') as inp:
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        input_ids_l = pickle.load(inp)
        label_l = pickle.load(inp)
        input_mask_l = pickle.load(inp)
        output_mask_l = pickle.load(inp)

    with open('data/test_data.txt', 'r', encoding='utf-8') as f:
        for test in f:
            flag = False
            test = test.strip()

            tokens = [i for i in test]
            tokens_cs = '[CLS]' + ' '.join(tokens) + '[SEP]'
            tokenized_text = tokenizer.tokenize(tokens_cs)
            input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
            input_mask = [1] * len(input_ids)
            while len(input_ids) < 512:
                input_ids.append(0)
                input_mask.append(0)
            output_mask = [1] * len(tokens)
            output_mask = [1] + output_mask + [1]
            while len(output_mask) < 512:
                output_mask.append(0)

            input_ids = torch.LongTensor(input_ids).view(1, -1)
            input_mask = torch.BoolTensor(input_mask).view(1, -1)
            output_mask = torch.BoolTensor(output_mask).view(1, -1)

            predict = model.infer(input_ids, input_mask, output_mask)[0]

            for i in range(len(test)):
                print(test[i], end='', file=output)
                print(id2tag[predict[i+1]], end=' ', file=label_output)
                if id2tag[predict[i+1]] in ['E', 'S']:
                    print(' ', end='', file=output)
            print(file=output)
            print(file=label_output)
    output.close()
    label_output.close()
