import torch
from torch.nn import Sequential
from torchvision.models import AlexNet
import torch
import torch.nn as nn
from torch.autograd import Function

from torch.hub import load_state_dict_from_url
import copy

from inspect import  getfullargspec
model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class ReverseLayerF(Function):
    # Forwards identity
    # Sends backward reversed gradients
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

copy_container = lambda container: Sequential(*[
            x.__class__(**{key: value for key, value in x.__dict__.items() if key in getfullargspec(x.__class__).args})
            for x in container])

class DANN(AlexNet):
    def __init__(self, num_classes, pretrained=False, num_domains=1,  **kwargs):

        # Initialize AlexNet
        super().__init__(**kwargs)
        self.domain_classifier = copy_container(self.classifier)

        # Load weights of AlexNet trained on ImageNet
        if pretrained:
            print("Loading AlexNet")
            state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                                  progress=True)
            self.load_state_dict(state_dict, strict=False)

            #todo vedere se ultimi pesi vanno considerati
            for i in [1,4,6]:
                self.domain_classifier[i].weight.data = self.classifier[i].weight.data.detach().clone().requires_grad_(True)
                self.domain_classifier[i].bias.data = self.classifier[i].bias.data.detach().clone().requires_grad_(True)


        # Initialize Discriminator's branch as copy of the structure of classifier
        # self.domain_classifier = copy.deepcopy(self.classifier)
        # self.domain_classifier.load_state_dict(self.classifier.state_dict(), strict=False)

        # Change final layers
        self.classifier[6].out_features = num_classes
        self.domain_classifier[6].out_features = num_domains

    def forward(self, x, alpha=False):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if alpha is None:
            x = self.classifier(x)
        else:
            x = ReverseLayerF.apply(x, alpha)
            x = self.discriminator(x)

        return x




if __name__ == '__main__':
    dann = DANN(pretrained=True, num_classes=7, num_domains=2)
    print(dann)
    # import torch
    #
    #
    # if type(value) == nn.Parameter:
    #     print(key, "is a parameter!"
    #
    #     getattr(torch.nn, dann.classifier[0].__repr__())
    #     prova = torch.nn.Sequential(
    #         *[x.__class__(**{key: value for key, value in x.__dict__.items()}) for x in dann.classifier])