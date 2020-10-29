import torch.nn as nn

from torch.autograd import Function


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        reverse_grad_output = grad_output.neg() * ctx.alpha
        return reverse_grad_output, None


class DomainModel(nn.Module):
    def __init__(self, config):
        super(DomainModel, self).__init__()
        self.config = config
        self.activation = nn.Tanh()
        self.pool = nn.Linear(self.config.bert_size, self.config.bert_size)

        self._domain_classification = nn.Sequential(nn.Dropout(p=self.config.dropout),
                                                    nn.Linear(self.config.bert_size,
                                                    self.config.dis_class_num))

    def forward(self, domain_input):
        grl_domain_input = ReverseLayerF.apply(domain_input, self.config.alpha)
        first_token_tensor = grl_domain_input[:, 0]
        pooled_output = self.pool(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        logits = self._domain_classification(pooled_output)
        return logits