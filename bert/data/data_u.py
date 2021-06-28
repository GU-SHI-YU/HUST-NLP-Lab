import codecs
from sklearn.model_selection import train_test_split
import pickle
import torch
from transformers import AlbertModel, BertTokenizer

INPUT_DATA = "train.txt"
SAVE_PATH = "./datasave.pkl"
id2tag = ['B', 'M', 'E', 'S']  # B：分词头部 M：分词词中 E：分词词尾 S：独立成词
tag2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
word2id = {}
id2word = []
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")


def get_list(input_str):
    """
    单个分词转换为tag序列
    :param input_str: 单个分词
    :return: tag序列
    """
    output_str = []
    if len(input_str) == 1:
        output_str.append(tag2id['S'])
    elif len(input_str) == 2:
        output_str = [tag2id['B'], tag2id['E']]
    else:
        M_num = len(input_str) - 2
        M_list = [tag2id['M']] * M_num
        output_str.append(tag2id['B'])
        output_str.extend(M_list)
        output_str.append(tag2id['E'])
    return output_str


def handle_data():
    """
    处理数据，并保存至savepath
    :return:
    """
    input_ids_l = []
    input_mask_l = []
    label_l = []
    output_mask_l = []
    line_num = 0
    with open(INPUT_DATA, 'r', encoding="utf-8") as ifp:
        for line in ifp:
            line_num = line_num + 1
            line = line.strip()
            if not line:
                continue
            words = line.split()
            sent = ''.join(words)
            tokens = [i for i in sent]
            label = []
            for item in words:
                label.extend(get_list(item))
            if len(tokens) > 512 - 2:
                tokens = tokens[: (512 - 2)]
                label = label[: (512 - 2)]
            tokens_cs = '[CLS]' + ' '.join(tokens) + '[SEP]'
            tokenized_text = tokenizer.tokenize(tokens_cs)
            input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
            input_mask = [1] * len(input_ids)
            label = [3] + label + [3]
            while len(input_ids) < 512:
                input_ids.append(0)
                input_mask.append(0)
            while len(label) < 512:
                label.append(-1)

            output_mask = [1] * len(tokens)
            output_mask = [1] + output_mask + [1]
            while len(output_mask) < 512:
                output_mask.append(0)

            assert len(input_ids) == 512
            assert len(input_mask) == 512
            assert len(label) == 512
            assert len(output_mask) == 512

            input_ids_l.append(input_ids)
            input_mask_l.append(input_mask)
            label_l.append(label)
            output_mask_l.append(output_mask)

    print(tokenizer.convert_ids_to_tokens(input_ids_l[0]))
    print(input_ids_l[0])
    print(input_mask_l[0])
    print(label_l[0])
    print(output_mask_l[0])
    with open(SAVE_PATH, 'wb') as outp:
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
        pickle.dump(input_ids_l, outp)
        pickle.dump(label_l, outp)
        pickle.dump(input_mask_l, outp)
        pickle.dump(output_mask_l, outp)


if __name__ == "__main__":
    handle_data()
