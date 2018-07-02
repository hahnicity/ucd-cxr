"""
init
~~~~

An extension of torch.nn.init and just provides convenience.

Usage:

    from cxrlib.init import kaiming_init

    model = MyModel()
    model.apply(kaiming_init)
"""
from torch.nn.init import kaiming_uniform_, xavier_uniform_


def kaiming_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        kaiming_uniform_(m.weight.data)
        if m.bias is not None:
            kaiming_uniform_(m.bias.data)


def xavier_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        xavier_uniform_(m.weight.data)
        if m.bias is not None:
            xavier_uniform_(m.bias.data)
