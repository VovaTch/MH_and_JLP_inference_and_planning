from __future__ import print_function
import numpy as np
import scipy as sp
import scipy.io
from scipy import special
from scipy import stats
import gtsam
import math
import damodel
import geomodel as geomodel_proj
import plotterdaac2d
import gaussianb_jlp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from JLP_planner import JLPPLannerPrimitives
from hybridb import HybridBelief
from lambdab_lg import LambdaBelief
from lambda_planner import LambdaPlannerPrimitives
import time
import json
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import models
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image


# Load Alexnet
alexnet = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19_bn', pretrained=True)
#alexnet.train().cuda()

def apply_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.train()

alexnet.eval()
alexnet.apply(apply_dropout)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# No dropout classification
def alexnet_classification(image, class_list=(921, 692, 737, 530, 804)):

    cropped_image = transform(image)
    cropped_image = torch.unsqueeze(cropped_image, 0)
    cropped_image = cropped_image.cuda()

    cls_out = alexnet(cropped_image)
    cls_out = torch.nn.functional.softmax(cls_out)

    cls_np_out = np.zeros((len(class_list), 1))
    for idx, cls in enumerate(class_list):
        cls_np_out[idx] = cls_out[0][cls].detach().cpu()

    cls_np_out = cls_np_out / np.sum(cls_np_out)

    return cls_np_out

# Output LG parameters from MC dropout
def MC_dropout_alexnet(image, number_of_dropouts=100, class_list=(921, 692, 737, 530, 804)):

    lg_expectation = np.zeros(len(class_list) - 1)
    lg_samples = np.zeros([number_of_dropouts, len(class_list) - 1])
    lg_covariance = np.zeros([len(class_list) - 1, len(class_list) - 1])

    for do_seed in range(number_of_dropouts):

        cropped_image = transform(image)
        cropped_image = torch.unsqueeze(cropped_image, 0)
        cropped_image = cropped_image.cuda()

        torch.manual_seed(do_seed)
        cls_out = alexnet(cropped_image)
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

    print(lg_expectation)
    print(lg_covariance)

    return lg_expectation, lg_covariance

# Find the corresponding depth image to the jpg
def depth_image_find(image_address):

    depth_address = image_address[0:-5]
    depth_address += '3.png'
    return depth_address

# Load json file
actions = ['forward','rotate_ccw','rotate_cw','backward','left','right']

scenario_file = open('../../../Office_001_1/annotations.json')
scenario_data = json.load(scenario_file)

detailed_GT_file = scipy.io.loadmat('../../../Office_001_1/image_structs.mat')

x_pos = list()
y_pos = list()
z_pos = list()
for location in detailed_GT_file['image_structs']['world_pos'][0]:
    x_pos.append(float(location[0]))
    y_pos.append(float(location[2]))
    z_pos.append(float(location[1]))

psi = list()
theta = list()
for rotation in detailed_GT_file['image_structs']['direction'][0]:
    psi.append(math.atan2(rotation[2], rotation[0]) * 180 / np.pi)
    theta.append(math.atan2(rotation[1], np.sqrt(rotation[2]**2 + rotation[0]**2)) * 180 / np.pi)

x_pos = np.array(x_pos)
y_pos = np.array(y_pos)
z_pos = np.array(z_pos)

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
plt.plot(x_pos, y_pos, z_pos, 'r*')
plt.plot(x_pos[0:1], y_pos[0:1], z_pos[0:1], 'b*')
plt.xlabel('x')
plt.ylabel('y')

plt.show()


current_image = '100110000010101.jpg'
current_depth = depth_image_find(current_image)

measurement_data = dict()

#for idx in range(15):

for current_image in scenario_data:

    measurement_data[current_image] = dict()
    measurement_data[current_image]['LG expectation'] = list()
    measurement_data[current_image]['LG covariance'] = list()
    measurement_data[current_image]['Angles'] = list()
    measurement_data[current_image]['Range'] = list()
    measurement_data[current_image]['DA'] = list()
    measurement_data[current_image]['CPV'] = list()

    action = actions[1]
    img_rgb_read = Image.open('../../../Office_001_1/jpg_rgb/' + str(current_image))
    img_depth_read = mpimg.imread('../../../Office_001_1/high_res_depth/' + str(current_depth))
    #plt.imshow(img_rgb_read)
    #plt.show()
    #plt.imshow(img_depth_read)
    #plt.show()

    image_size = img_depth_read.shape

    for bbox in scenario_data[current_image]['bounding_boxes']:

        # Computing angles and middle of bounding box for range
        Middle = ((bbox[2] - bbox[0]) / 2 + bbox[0], (bbox[3] - bbox[1]) / 2 + bbox[1])
        Angles = [- 100 * (Middle[0] / 1920 - 1 / 2), - 56.25 * (Middle[1] / 1080 - 1 / 2)]

        # Cropping the image
        # Cropping the image
        # cropped_image = img_rgb_read.crop((math.ceil(bbox[0]),
        #                                   math.ceil(bbox[1]),
        #                                   math.ceil(bbox[2]),
        #                                   math.ceil(bbox[3])))

        max_width = np.max((bbox[2] - bbox[0], bbox[3] - bbox[1]))

        cropped_image = img_rgb_read.crop((- math.ceil(max_width / 2) + math.ceil(Middle[0]),
                                           - math.ceil(max_width / 2) + math.ceil(Middle[1]),
                                           math.ceil(max_width / 2) + math.ceil(Middle[0]),
                                           math.ceil(max_width / 2) + math.ceil(Middle[1])))

        alexnet.train().cuda()
        exp_lg, cov_lg = MC_dropout_alexnet(cropped_image)
        alexnet.eval().cuda()
        cpv = alexnet_classification(cropped_image)
        cov_lg_flattened = list()
        for idx_1 in range(4):
            for idx_2 in range(idx_1, 4):
                cov_lg_flattened.append(cov_lg[idx_1, idx_2])

        #plt.show()

        # Cropping the depth image for geometric measurements
        cropped_depth = img_depth_read[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        measurement_data[current_image]['LG expectation'].append(exp_lg.tolist())
        measurement_data[current_image]['LG covariance'].append(cov_lg_flattened)
        measurement_data[current_image]['Angles'].append(Angles)
        measurement_data[current_image]['Range'].append(cropped_depth[int(Middle[1] - bbox[1]),
                                                                         int(Middle[0] - bbox[0])] * 100)
        measurement_data[current_image]['DA'].append([bbox[4], bbox[5]])
        measurement_data[current_image]['CPV'].append(cpv[:, 0].tolist())
        #plt.imshow(cropped_depth)
        print('GT class: ' + str(bbox[4]))
        # if int(bbox[4]) == 16:
        plt.imshow(cropped_image)
        plt.show()

        # Printing physical sizes based on RGBD images
        # print('Distance: ' + str(cropped_depth[int(Middle[1] - bbox[1]), int(Middle[0] - bbox[0])] * 100))
        # print('Middle: ' + str(Middle))
        # print('Psi: ' + str(Angles[0]))
        # print('Theta ' + str(Angles[1]))
        # print('LG expectation: ' + str(exp_lg))
        # print('LG covariance: \n' + str(cov_lg))


    #current_image = scenario_data[current_image][action]
    #current_depth = depth_image_find(current_image)
    #print(current_image)

with open('classification_data.json', 'w') as outfile:
    json.dump(measurement_data, outfile)
outfile.close()
scenario_file.close()