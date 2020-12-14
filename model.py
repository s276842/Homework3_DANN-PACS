from torchvision.models import AlexNet

class DANN(AlexNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)



if __name__ == '__main__':
    dann = DANN()
    # import torch
    #
    #
    # if type(value) == nn.Parameter:
    #     print(key, "is a parameter!"
    #
    #     getattr(torch.nn, dann.classifier[0].__repr__())
    #     prova = torch.nn.Sequential(
    #         *[x.__class__(**{key: value for key, value in x.__dict__.items()}) for x in dann.classifier])