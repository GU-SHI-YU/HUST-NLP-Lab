import sys

with open('data/cws_res.txt', 'r', encoding='utf8') as cws:
    with open('data/ner_res.txt', 'r', encoding='utf8') as ner:
        with open('data/test_data.txt', 'r', encoding='utf8') as test:
            with open('cws_result.txt', 'w', encoding='utf8') as out:
                for cws_label, ner_label, line in zip(cws, ner, test):
                    cws_label = cws_label.split(" ")
                    ner_label = ner_label.split(" ")
                    length = len(line)
                    assert len(cws_label) == len(ner_label)
                    assert len(cws_label) == length
                    if len(sys.argv) > 2 and sys.argv[1] == 'on':
                        i = 0
                        while i < length:
                            if line[i] == '《':
                                print('title')
                                j = i + 1
                                i = i + 1
                                while line[j] != '》':
                                    j = j + 1
                                j = j - 1
                                if j - i == 0:
                                    cws_label[i] = 'S'
                                    break
                                cws_label[i] = 'B'
                                while i < j:
                                    i = i + 1
                                    cws_label[i] = 'M'
                                cws_label[j] = 'E'
                            if ner_label[i][0] == 'B':
                                # print('encounter entity')
                                j = i
                                needReplace = False
                                count = 0
                                while ner_label[j][0] != 'E':
                                    if cws_label[j] == 'E':
                                        count = count + 1
                                    if count >= 2:
                                        needReplace = True
                                        break
                                    j = j + 1
                                if cws_label[i] == 'E' and count == 1:
                                    needReplace = True
                                if needReplace:
                                    print('replace')
                                    if cws_label[i] == 'M' or cws_label[i] == 'E':
                                        if cws_label[i - 1] == 'B':
                                            cws_label[i - 1] = 'S'
                                        elif cws_label[i - 1] == 'M':
                                            cws_label[i - 1] = 'E'
                                        cws_label[i] == 'B'
                                    elif cws_label[i] == 'S':
                                        cws_label[i] = 'B'
                                    i = i + 1
                                    while ner_label[i][0] != 'E':
                                        cws_label[i] = 'M'
                                        i = i + 1
                                    if cws_label[i] == 'B' or cws_label[i] == 'M':
                                        if cws_label[i + 1] == 'M':
                                            cws_label[i + 1] = 'B'
                                        elif cws_label[i + 1] == 'E':
                                            cws_label[i + 1] = 'S'
                                        cws_label[i] = 'E'
                                    elif cws_label[i] == 'S':
                                        cws_label[i] == 'E'
                                else:
                                    i = j
                            i = i + 1

                    # print(' '.join(cws_label))
                    # exit(0)
                    for i in range(len(line)):
                        print(line[i], end='', file=out)
                        if cws_label[i] in ['E', 'S']:
                            print(' ', end='', file=out)
                    # print(file=out)
