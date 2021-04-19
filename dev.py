import numpy as np
from data import preprocess, transform, build_up_for_auxiliary_model
from model import classifier, Seq2Seq
import torch
import pickle
import random

random.seed(0)

w2id = pickle.load(open('e2e-dataset/w2id.pkl', 'rb'))
id2w = pickle.load(open('e2e-dataset/id2w.pkl', 'rb'))

# mr, mr_lengths = transform(w2id, 'e2e-dataset/devset_mr.txt')
ref, ref_lengths = transform(w2id, 'e2e-dataset/devset_ref.txt')
binary_representation = build_up_for_auxiliary_model('e2e-dataset/devset.csv', False)

embedding_size = 50
batch_size = 20
vocab_size = len(w2id)
value_size = binary_representation.shape[1]

data_size = len(ref)
iters = data_size//batch_size
if iters * batch_size < data_size:
    iters += 1

# clf = classifier(vocab_size, embedding_size, value_size)
# for k in range(20):
#     clf.load_state_dict(torch.load('checkpoint/'+str(k+1)+'-parameter.pkl'))
#     loss = 0
#     for i in range(iters):
#         start = i*batch_size
#         end = min((i+1)*batch_size, data_size)
#         y = clf.forward(torch.LongTensor(ref[start:end]), torch.LongTensor(ref_lengths[start:end]))
#         y[y <= 0.5] = 0
#         y[y > 0.5] = 1
#         t_tgt = binary_representation[start:end]
#         loss += np.sum(abs(y.detach().numpy()-t_tgt))
#     print(k+1, loss)

