import pandas as pd
import numpy as np
import re
import pickle


def preprocess(file_name):
    data = pd.read_csv(file_name).values
    if data.shape[1] == 1:
        f1 = open(file_name.split('.')[0]+'_mr.txt', 'w', encoding='utf8')
        records = []
        for mr in data:
            record = dict()
            mr_process = ''
            tuples = mr[0].lower().split(',')
            for tp in tuples:
                tp = tp.strip().split('[')
                if tp[0] == 'name':
                    record['name'] = tp[1][:-1]
                    mr_process += 'name X-name '
                elif tp[0] == 'near':
                    record['near'] = tp[1][:-1]
                    mr_process += 'near X-near '
                else:
                    mr_process = mr_process + tp[0] + ' ' + tp[1][:-1] + ' '
            f1.write(mr_process)
            f1.write('\n')
            records.append(record)
        f1.close()
        return records
    else:
        f1 = open(file_name.split('.')[0]+'_mr.txt', 'w', encoding='utf8')
        f2 = open(file_name.split('.')[0]+'_ref.txt', 'w', encoding='utf8')
        records = []
        for pair in data:
            mr = pair[0]
            mr_process = ''
            record = dict()
            ref = pair[1].lower()
            tuples = mr.lower().split(',')
            for tp in tuples:
                tp = tp.strip().split('[')
                if tp[0] == 'name':
                    record['name'] = tp[1][:-1]
                    ref = re.sub(record['name'], 'X-name', ref)
                    mr_process += 'name X-name '
                elif tp[0] == 'near':
                    record['near'] = tp[1][:-1]
                    ref = re.sub(record['near'], 'X-near', ref)
                    mr_process += 'near X-near '
                else:
                    mr_process = mr_process + tp[0] + ' ' + tp[1][:-1] + ' '
            ref = ref.replace('.', ' . ')
            ref = ref.replace(',', ' ,')
            f1.write(mr_process)
            f1.write('\n')
            f2.write('$SOF$ '+ref)
            f2.write('\n')
            records.append(record)
        f1.close()
        f2.close()
        return records


def create_dict(mr_fn, ref_fn):
    word_set = set()
    w2id = dict()
    id2w = dict()
    w2id['$EOF$'] = 0
    id2w[0] = '$EOF$'
    word_set.add('$EOF$')
    w2id['$SOF$'] = 1
    id2w[1] = '$SOF$'
    word_set.add('$SOF$')
    word_id = 2
    with open(mr_fn, encoding='utf8') as f:
        mrs = f.readlines()
        for mr in mrs:
            words = mr.split()
            for w in words:
                if w not in word_set:
                    w2id[w] = word_id
                    id2w[word_id] = w
                    word_id += 1
                    word_set.add(w)

    with open(ref_fn, encoding='utf8') as g:
        refs = g.readlines()
        for ref in refs:
            words = ref.split()
            for w in words:
                if w not in word_set:
                    w2id[w] = word_id
                    id2w[word_id] = w
                    word_id += 1
                    word_set.add(w)

    pickle.dump(w2id, open('e2e-dataset/w2id.pkl', 'wb'))
    pickle.dump(id2w, open('e2e-dataset/id2w.pkl', 'wb'))
    return w2id, id2w


def transform(word_dict, filename):
    lengths = []
    with open(filename, encoding='utf8') as f:
        lines = f.readlines()
        for sent in lines:
            length = len(sent.split())
            lengths.append(length)
        max_sen_length = max(lengths)
        to_input = np.zeros((len(lines), max_sen_length))
        for i in range(len(lines)):
            sent = lines[i].split()
            j = 0
            for w in sent:
                if not word_dict.get(w, 0) == 0:
                    to_input[i][j] = word_dict[w]
                    j += 1
            lengths[i] = j
    return to_input, np.array(lengths)


def data_analyse(file_name, attrs):
    data = pd.read_csv(file_name).values
    values = [set() for _ in range(len(attrs))]
    for mr in data:
        s = mr[0].lower()
        pairs = s.split(",")
        for tp in pairs:
            tup = tp.strip().split("[")
            att = tup[0]
            x = attrs.get(att, -1)
            if not x < 0:
                values[x].add(tup[1][:-1])

    # for i in attrs.values():
    #     print(list(attrs.keys())[i], end=',')
    #     print(len(values[i]), end='--')
    #     print(values[i])

    return values


def build_up_for_auxiliary_model(filename, is_train):
    attributes = {'eattype': 0, 'food': 1, 'familyfriendly': 2, 'customer rating': 3, 'area': 4, 'pricerange': 5}
    if is_train:
        att_value = data_analyse(filename, attributes)
        i = 0
        v2id = dict()
        for s in attributes.keys():
            for v in att_value[attributes[s]]:
                v2id[s + '-' + v] = i
                i += 1
        pickle.dump(v2id, open('e2e-dataset/v2id.pkl', 'wb'))
    else:
        v2id = pickle.load(open('e2e-dataset/v2id.pkl', 'rb'))

    data = pd.read_csv(filename).values
    labels = []
    for mr in data:
        label = np.zeros(len(v2id))
        s = mr[0].lower()
        pairs = s.split(',')
        for tp in pairs:
            tup = tp.strip().split('[')
            att = tup[0]
            if att in {'name', 'near'}:
                continue
            value = tup[1][:-1]
            x = v2id.get(att+'-'+value, -1)
            if not x < 0:
                label[x] = 1
        labels.append(label)
    return np.stack(labels)


# preprocess('e2e-dataset/trainset.csv')
# preprocess('e2e-dataset/devset.csv')
# preprocess('e2e-dataset/testset.csv')
# build_up_for_auxiliary_model('e2e-dataset/trainset.csv', False)
