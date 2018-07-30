import torch
import torchvision.models as models
from matplotlib import pyplot as plt

def plot_kernels(tensor, num_cols=6):
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    if not tensor.shape[-1]==3:
        raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    
  
#vgg = models.vgg16(pretrained=True)
from model import DenseNet121
model = DenseNet121(14)
model = torch.nn.DataParallel(model)

ckpt = torch.load("../chexnet/model.pth.tar",map_location=lambda storage, loc: storage)
model.load_state_dict(ckpt['state_dict'])

mm = model.double()

filters = mm.modules
print(type(filters))
body_model = [i for i in mm.children()][0]
print(type(body_model))
layer1 = body_model
tensor = conv0.weight.data.numpy()
plot_kernels(tensor)
