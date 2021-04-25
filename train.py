import numpy as np
from data import preprocess, create_dict, transform, build_up_for_auxiliary_model
from model import classifier, Seq2Seq
import torch.optim as opt
import torch
import pickle

# records = preprocess('e2e-dataset/trainset.csv')
# w2id, id2w = create_dict('e2e-dataset/trainset_mr.txt', 'e2e-dataset/trainset_ref.txt')
w2id = pickle.load(open('e2e-dataset/w2id.pkl', 'rb'))
id2w = pickle.load(open('e2e-dataset/id2w.pkl', 'rb'))

mr, mr_lengths = transform(w2id, 'e2e-dataset/trainset_mr.txt')
ref, ref_lengths = transform(w2id, 'e2e-dataset/trainset_ref.txt')
binary_representation = build_up_for_auxiliary_model('e2e-dataset/trainset.csv', False)

embedding_size = 50
hidden_size = 128
batch_size = 20

vocab_size = len(w2id)
data_size = len(mr)
value_size = binary_representation.shape[1]

iters = data_size // batch_size
# if iters * batch_size < data_size:
#     iters += 1


def train_binary_predictor():
    clf = classifier(vocab_size, embedding_size, hidden_size, value_size)
    optimizer = opt.Adam(clf.parameters(), lr=1e-3)

    num_epoch = 20
    for i in range(num_epoch):
        print('epoch {}/{}'.format(i + 1, num_epoch))
        shuffle_indices = np.random.permutation(np.arange(data_size))
        # shuffle_indices = np.arange(data_size)
        txt_ = ref[shuffle_indices]
        lengths_ = ref_lengths[shuffle_indices]
        tgt_ = binary_representation[shuffle_indices]

        for j in range(iters):
            start = j * batch_size
            end = min(data_size, (j + 1) * batch_size)
            y = clf.forward(torch.LongTensor(txt_[start:end]), torch.LongTensor(lengths_[start:end]))
            loss = -torch.sum(torch.mul(torch.log(y), torch.LongTensor(tgt_[start:end]))) \
                   - torch.sum(torch.mul(torch.log(1-y), torch.LongTensor(1 - tgt_[start:end])))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(loss)

        torch.save(clf.state_dict(), 'checkpoint/' + str(101+i) + '-parameter.pkl')


def train_txt_generator():
    seq2seq = Seq2Seq(vocab_size, hidden_size, embedding_size)
    optimizer = opt.Adam(seq2seq.parameters(), lr=5e-4)

    num_epoch = 1

    for i in range(num_epoch):
        print('epoch {}/{}'.format(i + 1, num_epoch))
        shuffle_indices = np.random.permutation(np.arange(data_size))
        mr_ = mr[shuffle_indices]
        lengths_ = mr_lengths[shuffle_indices]
        ref_ = ref[shuffle_indices]

        for j in range(1):
            start = j * batch_size
            end = min(data_size, (j + 1) * batch_size)
            y = seq2seq.forward(torch.LongTensor(mr_[start:end]), torch.LongTensor(ref_[start:end]),
                                torch.LongTensor(lengths_[start:end]))
            ref_gt = np.array(ref[start:end], dtype=int)
            tgt = torch.tensor(np.eye(vocab_size)[ref_gt])
            loss = -torch.sum(torch.mul(torch.log(y)[:, :-1, :], tgt[:, 1:, :])) \
                   - torch.sum(torch.mul(torch.log(1-y)[:, :-1, :], 1-tgt[:, 1:, :]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)

            if (j+1) % 200 == 0:
                torch.save(seq2seq.state_dict(), 'checkpoint/s2s-' + str(j + 1) + '-parameter.pkl')
                # txt = seq2seq.generate(torch.LongTensor(mr[0]), torch.LongTensor([mr_lengths[0]]), 5)[0].detach().numpy()
                # for index in txt:
                #     print(id2w[index], end=' ')
                # print('\n')


train_binary_predictor()
# train_txt_generator()
