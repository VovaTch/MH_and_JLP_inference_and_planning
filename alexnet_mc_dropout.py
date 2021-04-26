import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import models
from torch.autograd import Variable
from torchvision import transforms
import numpy as np

class MCDropoutClassifier:

    def __init__(self):

        # Load Alexnet
        self.alexnet = models.alexnet(pretrained=True)
        self.alexnet.train().cuda()
        self.transform = transforms.Compose([  # [1]
            transforms.Resize(224),  # [2]
            transforms.CenterCrop(224),  # [3]
            transforms.ToTensor(),  # [4]
            transforms.Normalize(  # [5]
                mean=[0.485, 0.456, 0.406],  # [6]
                std=[0.229, 0.224, 0.225]  # [7]
            )])

    # Output LG parameters from MC dropout
    def MC_dropout_alexnet(self, image, number_of_dropouts=20, class_list=(921, 692, 737, 530, 804)):

        lg_expectation = np.zeros(len(class_list) - 1)
        lg_samples = np.zeros([number_of_dropouts, len(class_list) - 1])
        lg_covariance = np.zeros([len(class_list) - 1, len(class_list) - 1])

        for do_seed in range(number_of_dropouts):

            cropped_image = self.transform(image)
            cropped_image = torch.unsqueeze(cropped_image, 0)
            cropped_image = cropped_image.cuda()

            torch.manual_seed(do_seed)
            cls_out = self.alexnet(cropped_image)
            cls_out = torch.nn.functional.softmax(cls_out)

            cls_list_val = list()
            for i in class_list:
                cls_list_val.append(cls_out[0, i].item())
            sum_val = sum(cls_list_val)
            for i in range(len(class_list)):
                cls_list_val[i] = cls_list_val[i] / sum_val

            lg_samples[do_seed, :] = np.log(cls_list_val[0:-1]) - \
                                     np.log(cls_list_val[-1]) * np.ones(len(class_list) - 1)
            lg_expectation += lg_samples[do_seed, :] / number_of_dropouts

        for do_seed in range(number_of_dropouts):
            lg_covariance += np.outer(lg_samples[do_seed, :] - lg_expectation, lg_samples[do_seed, :] - lg_expectation) \
                             / (number_of_dropouts - 1)

        return lg_expectation, lg_covariance