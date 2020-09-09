from math import isclose
import torch


class AliasMultinomial(torch.nn.Module):
    '''Alias sampling method to speedup multinomial sampling

    The alias method treats multinomial sampling as a combination of uniform sampling and
    bernoulli sampling. It achieves significant acceleration when repeatedly sampling from
    the save multinomial distribution.

    Attributes:
        - probs: the probability density of desired multinomial distribution

    Refs:
        - https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    '''
    def __init__(self, probs):
        super(AliasMultinomial, self).__init__()

        assert isclose(probs.sum().item(), 1), 'The noise distribution must sum to 1'
        cpu_probs = probs.cpu()
        K = len(probs)

        # such a name helps to avoid the namespace check for nn.Module
        self_prob = [0] * K
        self_alias = [0] * K

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for idx, prob in enumerate(cpu_probs):
            self_prob[idx] = K*prob
            if self_prob[idx] < 1.0:
                smaller.append(idx)
            else:
                larger.append(idx)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        # 通过拼凑，将各个类别都凑为1

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self_alias[small] = large
            self_prob[large] = (self_prob[large] - 1.0) + self_prob[small]

            if self_prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        # self_prob 得到原类型的概率
        # Alias 第二种类型

        for last_one in smaller+larger:
            self_prob[last_one] = 1

        self.register_buffer('prob', torch.Tensor(self_prob))

        # self_prob 中第四个和第五个元素概率为1，self_alias表格中，对应的是0，而其余的均不是0
        # 参考 https://shomy.top/2017/05/09/alias-method-sampling/
        self.register_buffer('alias', torch.LongTensor(self_alias))

    def draw(self, *size):
        """Draw N samples from multinomial

        Args:
            - size: the output size of samples
        """
        max_value = self.alias.size(0)   # max_value应该是 V 吧
        # size: 负样本的个数，每一个元算的value是0到V-1
        kk = self.alias.new(*size).random_(0, max_value).long().view(-1)  # *size (bsz, 1, noise_num)
        prob = self.prob[kk]  # kk是token的index, prob[kk]取到这个token的概率，反应了noise distribution
        alias = self.alias[kk]
        # b is whether a random number is smaller than prob, 小于输出1
        b = torch.bernoulli(prob).long()  # 如果随机数小于prob， 则取该列本身对应的token，否也取填补到这列的token
        oq = kk.mul(b)
        oj = alias.mul(1 - b)  # 随机数大于prob， b为0， 1-b为1，取alias[i]

        return (oq + oj).view(size)

