from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

import os
class PACS():
    def __init__(self, root, transform=None, batch_size=64, num_workers=0):

        # Load all four datasets with ImageFolder
        self.__art_dataset = ImageFolder(os.path.join(root,'art_painting'), transform=transform)
        self.__cartoon_dataset = ImageFolder(os.path.join(root,'cartoon'), transform=transform)
        self.__photo_dataset = ImageFolder(os.path.join(root,'photo'), transform=transform)
        self.__sketch_dataset = ImageFolder(os.path.join(root,'sketch'), transform=transform)

        # Create dataloader for each dataset
        self.__art_dataloader = DataLoader(self.__art_dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=num_workers, drop_last=True)
        self.__cartoon_dataloader = DataLoader(self.__cartoon_dataset, batch_size=batch_size, shuffle=True,
                                            num_workers=num_workers, drop_last=True)
        self.__photo_dataloader = DataLoader(self.__photo_dataset, batch_size=batch_size, shuffle=True,
                                            num_workers=num_workers, drop_last=True)
        self.__sketch_dataloader = DataLoader(self.__sketch_dataset, batch_size=batch_size, shuffle=True,
                                            num_workers=num_workers, drop_last=True)

        # Classes (same for all dataset)
        self.classes = self.__art_dataset.classes

        # Dictionary to access the dataset/dataloaders
        self.dic = {'p':self.__photo_dataloader, 'photo':self.__photo_dataloader,
                    'a':self.__art_dataloader, 'art':self.__art_dataloader,
                    'c':self.__cartoon_dataloader, 'cartoon':self.__cartoon_dataloader,
                    's':self.__sketch_dataloader, 'sketch':self.__sketch_dataloader}



    def __getitem__(self, item):
        if type(item) is str:
            return self.dic.get(item[0].lower()).dataset
        if type(item) is tuple:
            try:
                return self.dic.get(item[0].lower()).dataset.__getitem__(item[1])
            except:
                pass

    def get_next_batch(self, cat):
        try:
            return next(iter(self.dic.get(cat.lower())))
        except:
            pass


    # def load(self, cat):
    #     for batch in self.dic[cat.lower()]:
    #         yield batch

    def __next__(self):
        try:
            return self.__curr_iter.__next__()
        except:
            return []

    def __iter__(self):
        try:
            return self.__curr_iter.__iter__()
        except:
            return []

    def __call__(self, cat):
        self.__curr_iter = self.dic.get(cat.lower())
        return self


if __name__ == '__main__':
    import torchvision
    from torchvision import transforms
    imgnet_mean, imgnet_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    # Define transforms for training phase
    train_transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(imgnet_mean, imgnet_std)])


    pacs = PACS('Homework3-PACS\PACS', transform=train_transform, batch_size=512, num_workers=4)
    # print(pacs['A',-1])

    print(pacs.get_next_batch('art')[1])
    import matplotlib.pyplot as plt
    for batch in pacs('cartoon'):
        plt.imshow(batch[0][0].permute(1,2,0))
        plt.show()
        print(pacs.classes[batch[1][0]])