import torch
from torch.autograd import Variable
from torchvision import utils
import numpy as np
import random
import time
import os

from setup_mnist import MNIST  # MNISTModel
from models import LeNetpp
from L2_attack import CarliniL2
from Li_attack import CarliniLi

import logging
logging.basicConfig(filename='result.log',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s -  %(message)s')
logging.disable(logging.DEBUG)


def show(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#" + "#" * 100
    img = (img.flatten() + .5) * 3
    if len(img) != 784:
        return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i * 28:i * 28 + 28]]))


def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1, 1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            for j in seq:
                if (j == np.argmax(data.test_labels[start + i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start + i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start + i])
            targets.append(data.test_labels[start + i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets


def distortion(a, b, norm_type='l2'):
    if norm_type == 'l1':
        return np.sum(np.abs(a - b))

    elif norm_type == 'l2':
        return np.sum((a - b)**2)**.5

    elif norm_type == 'li':
        return np.amax(np.abs(a - b))
    else:
        return None


def creat_attack(model, norm_type='l2'):
    if norm_type == 'l1':
        return None

    elif norm_type == 'l2':
        return CarliniL2(model, (9, 1, 28, 28), max_iterations=1000, confidence=0)

    elif norm_type == 'li':
        return CarliniLi(model, (1, 1, 28, 28))
    else:
        return None


def run():
    repeat_num = 1
    start_offset = 3
    adv_root = './adv'
    config = {
        'softmax_li': ('./softmax.pth', False, False, start_offset, False, 'li'),
        'softmax_l2': ('./softmax.pth', False, False, start_offset, False, 'l2')
    }

    result_dict = dict()
    for config_name, (model_path, is_variational, is_sphere, start, inception, norm_type) in config.items():
        distortion_list = []
        for j in range(repeat_num):
            data = MNIST()
            model = LeNetpp()
            model.cuda()
            model.load_state_dict(torch.load(model_path))

            attack = creat_attack(model, norm_type=norm_type)
            inputs, targets = generate_data(data, samples=1, targeted=True,
                                            start=start, inception=inception)

            save_root = os.path.join(
                adv_root, config_name + '-{}-start{}-incep{}-{}-adv.npy'.format(j, start, inception, norm_type))
            try:
                adv = np.load(save_root)
            except:
                print('attack begin')
                timestart = time.time()
                adv = attack.attack(inputs, targets)
                np.save(save_root, adv)
                timeend = time.time()
                print("Took", timeend - timestart,
                      "seconds to run", len(inputs), "samples.")

            li_distortion = 0.0
            l2_distortion = 0.0
            success_attack_num = 0.0
            for i in range(len(adv)):
                print("Valid:")
                show(inputs[i])
                print("Adversarial:")
                show(adv[i])

                li_dist = distortion(
                    adv[i], inputs[i], norm_type='li')
                l2_dist = distortion(
                    adv[i], inputs[i], norm_type='l2')
                print(
                    config_name + " {:.4f} th li: {:.4f}, l2: {:.4f}".format(i, li_dist, l2_dist))
                logging.debug(
                    config_name + " {:.4f} th li: {:.4f}, l2: {:.4f}".format(i, li_dist, l2_dist))
                if li_dist > 0 or l2_dist > 0:
                    success_attack_num += 1
                li_distortion += li_dist
                l2_distortion += l2_dist

            if success_attack_num > 0:
                li_mean = li_distortion / success_attack_num
                l2_mean = l2_distortion / success_attack_num
            else:
                li_mean = 'fail attack'
                l2_mean = 'fail attack'

            success_attack_rate = success_attack_num / len(adv)
            print('Mean li : {:.4f} l2: {:.4f} success rate {:.0f} %'.format(li_mean, l2_mean,
                                                                             success_attack_rate * 100))
            logging.info(
                config_name + ' Mean li : {:.4f} l2: {:.4f} success rate {:.0f} %'.format(li_mean, l2_mean,
                                                                                          success_attack_rate * 100))
            distortion_list.append([li_mean, l2_mean, success_attack_rate])

        result_dict[config_name] = tuple(distortion_list)



if __name__ == "__main__":
    run()