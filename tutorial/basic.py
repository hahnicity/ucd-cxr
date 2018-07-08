import torch
from torchvision.models import resnet50
from torchvision.transforms import Compose, Normalize
from torch.optim import SGD
from torch.nn import MSELoss

from cxrlib.read_data import RandomDataset

# Utilize ResNet50. Put it on the CPU
model = resnet50().cpu()
model.fc = torch.nn.Linear(2048, 1)

# number of epochs to run our model for
n_eps = 1

# Loss function to compute gradients in backpropagation
criterion = MSELoss()

# Optimizer function modifies model parameters based on the computed gradient
optimizer = SGD(model.parameters(), lr=0.01)

# Make sure we normalize all images based on pixel mean and std of ImageNet dataset
normalize = Normalize([0.485, 0.456, 0.406],
                      [0.229, 0.224, 0.225])

# Generate data
dataset = RandomDataset(transform=Compose([normalize]))

# DataLoader is a wrapping class that ensures we can process our data in mini-batches
loader = torch.utils.data.DataLoader(dataset, batch_size=16)
print("Training Start:")
# Run for a pre-specified number of epochs
for ep in range(n_eps):
    # Iterate over each mini-batch
    for inp, target in loader:
        # Cast the target and image input as Variable. This allows pytorch to run
        # meaningful computations on the data and compute a gradient when finished
        target = torch.autograd.Variable(target)
        inp = torch.autograd.Variable(inp)

        # Input our image into the model to get an output
        out = model(inp)

        # Set all gradients to 0 so we do not bias model from previous mini-batch
        optimizer.zero_grad()

        # Compute gradient
        loss = criterion(out, target)
        loss.backward()

        # Update model parameters
        optimizer.step()

        print("updating network")
    print("end epoch {}".format(ep))
