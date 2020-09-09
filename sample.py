"""A minimal sample script for illustration of basic usage of NCE module"""

import torch
from nce import IndexLinear

class_freq = [0, 2, 2, 3, 4, 5, 6]  # an unigram class probability
freq_count = torch.FloatTensor(class_freq)
print("total counts for all tokens:", freq_count.sum())
noise = freq_count / freq_count.sum()

# IndexLinear 继承了NCELoss 类
nce_linear = IndexLinear(
    embedding_dim=100,  # input dim
    num_classes=300000,  # output dim
    noise=noise,
)

# 这里 input 假装是经过了 embedding之后的
input = torch.Tensor(200, 100)  # [batch, emb_dim]
# target中这里是ones， 但是我们的task中应该是 对应的正确的token的id
target = torch.ones(200, 1).long()  # [batch, 1]
# training mode
loss = nce_linear(target, input).mean()
print(loss.item())

# evaluation mode for fast probability computation
nce_linear.eval()
prob = nce_linear(target, input).mean()
print(prob.item())
