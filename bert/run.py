import pickle
import logging
import argparse
import os
from re import L
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from model import CWS

def get_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=768)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--model', type=str, default='hfl/chinese-roberta-wwm-ext')
    return parser.parse_args()


def set_logger():
    log_file = os.path.join('save', 'log.txt')
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m%d %H:%M:%S',
        filename=log_file,
        filemode='w',
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def entity_split(x, y, id2tag, entities, cur):
    start, end = -1, -1
    for j in range(len(x)):
        if id2tag[y[j]] == 'B':
            start = cur + j
        elif id2tag[y[j]] == 'M' and start != -1:
            continue
        elif id2tag[y[j]] == 'E' and start != -1:
            end = cur + j
            entities.add((start, end))
            start, end = -1, -1
        elif id2tag[y[j]] == 'S':
            entities.add((cur + j, cur + j))
            start, end = -1, -1
        else:
            start, end = -1, -1


def main(args):
    use_cuda = args.cuda and torch.cuda.is_available()

    with open('data/datasave.pkl', 'rb') as inp:
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        input_ids_l = pickle.load(inp)
        label_l = pickle.load(inp)
        input_mask_l = pickle.load(inp)
        output_mask_l = pickle.load(inp)

    model = CWS(0, tag2id, args.embedding_dim, args.hidden_dim)
    if use_cuda:
        model = model.cuda()
    for name, param in model.named_parameters():
        logging.debug('%s: %s, require_grad=%s' % (name, str(param.shape), str(param.requires_grad)))

    optimizer = Adam(model.parameters(), lr=args.lr)

    random_order = list(range(len(input_ids_l)))
    np.random.shuffle(random_order)
    input_ids_l = [input_ids_l[i] for i in random_order]
    input_mask_l = [input_mask_l[i] for i in random_order]
    label_l = [label_l[i] for i in random_order]
    output_mask_l = [output_mask_l[i] for i in random_order]

    data_size = len(input_ids_l)
    test_size = data_size // 10

    test_input_ids = torch.LongTensor(input_ids_l[:test_size])
    train_input_ids = torch.LongTensor(input_ids_l[test_size:])
    test_input_mask = torch.BoolTensor(input_mask_l[:test_size])
    train_input_mask = torch.BoolTensor(input_mask_l[test_size:])
    test_label = torch.LongTensor(label_l[:test_size])
    train_label = torch.LongTensor(label_l[test_size:])
    test_output_mask = torch.BoolTensor(output_mask_l[:test_size])
    train_output_mask = torch.BoolTensor(output_mask_l[test_size:])

    train_dataset = TensorDataset(train_input_ids, train_input_mask, train_label, train_output_mask)
    test_dataset = TensorDataset(test_input_ids, test_input_mask, test_label, test_output_mask)

    train_data = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=6
    )

    test_data = DataLoader(
        dataset=test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=6
    )

    for epoch in range(args.max_epoch):
        step = 0
        log = []
        for input_ids, input_mask, label, output_mask in train_data:
            if use_cuda:
                input_ids = input_ids.cuda()
                label = label.cuda()
                input_mask = input_mask.cuda()
                output_mask = output_mask.cuda()
            # forward
            loss = model(input_ids, label, input_mask, output_mask)
            log.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % 100 == 0:
                logging.debug('epoch %d-step %d loss: %f' % (epoch, step, sum(log)/len(log)))
                log = []

        # test
        entity_predict = set()
        entity_label = set()
        with torch.no_grad():
            model.eval()
            cur = 0
            for input_ids, input_mask, label, output_mask in test_data:
                if use_cuda:
                    input_ids = input_ids.cuda()
                    label = label.cuda()
                    input_mask = input_mask.cuda()
                    output_mask = output_mask.cuda()
                predict = model.infer(input_ids, input_mask, output_mask)

                for i in range(len(predict)):
                    length = len([input_ids[i][j] for j in input_mask[i] if j == 1])
                    entity_split(input_ids[i, :length],
                                 predict[i],
                                 id2tag,
                                 entity_predict,
                                 cur)
                    entity_split(input_ids[i, :length],
                                 label[i, :length],
                                 id2tag,
                                 entity_label,
                                 cur)
                    cur += length

            right_predict = [i for i in entity_predict if i in entity_label]
            if len(right_predict) != 0:
                precision = float(len(right_predict)) / len(entity_predict)
                recall = float(len(right_predict)) / len(entity_label)
                logging.info("precision: %f" % precision)
                logging.info("recall: %f" % recall)
                logging.info("f1score: %f" % ((2 * precision * recall) / (precision + recall)))
            else:
                logging.info("precision: 0")
                logging.info("recall: 0")
                logging.info("f1score: 0")
            model.train()

        path_name = "./save/model_epoch" + str(epoch) + ".pkl"
        torch.save(model, path_name)
        logging.info("model has been saved in  %s" % path_name)


if __name__ == '__main__':
    set_logger()
    main(get_param())
