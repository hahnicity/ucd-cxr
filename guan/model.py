from torch import nn
import torchvision


def get_resnet():
    model = torchvision.models.resnet50(pretrained=True)
    # XXX need to figure out how many input features the linear layer will
    # have
    model.classifier = nn.Sequential(
        nn.Linear(XXX, 14),
        nn.Sigmoid()
    )
    return model


def main():
    classifier = get_resnet()
    pass


if __name__ == "__main__":
    main()
