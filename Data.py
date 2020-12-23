from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
import numpy as np
from torch import FloatTensor, IntTensor
from numpy.random import permutation
from numpy import zeros

def get_labeled_data(number_of_labeled_samples=100, balanced=True):
    train_dataset = datasets.MNIST('dataset/', train=True, transform=transforms.ToTensor(), download=True) # ToTensor is used to normalize or scales the features (by getting mean values), it takes (Height x Weight x Channel) in the range [0,255] to a tensor in the shape of ( Chanel x Height x Weight) in the range of [0.0, 1.0]
    length_of_dataset = train_dataset.__len__() # get length of train dataset
    random_indices = permutation(length_of_dataset) # get array of length equals to given number of samples that contains random indices
    samples_per_class = int (number_of_labeled_samples / 10) # to have a balanced number of samples for each digit class
    samples_per_class_list = zeros(10) # to keep in mind how much images we have for each digit
    images = []
    labels = []
    counter = 0
    for index in range(length_of_dataset):
        image,label = train_dataset.__getitem__(random_indices[index])
        # balanced chunk of digits per class (for example, we can have 10 randomized images per class to get total of 100 images for all classes combined, each digit(class) has 10 random images)
        if balanced:
            if all(class_ == samples_per_class for class_ in samples_per_class_list): # break if we've reached maximum images per each class
                break;
            if samples_per_class_list[label] < samples_per_class:
                images.append(np.array(image))
                labels.append(label)
                samples_per_class_list[label] += 1
        else : # if not balanced
            if counter == number_of_labeled_samples:
                break
            images.append(np.array(image))
            labels.append(label)
            counter += 1
        labeled_data = TensorDataset(FloatTensor(images), IntTensor(labels)) # image should be as Float tensors, and labels should be as Int Tensors
        repeated_times = (length_of_dataset // labeled_data.tensors[0].shape[0]) # labeled tensors will be repeated x times in order to reach the length of the normal dataset, repeated dataset will be used as minibatches
        t1, t2 = labeled_data.tensors[0].clone(), labeled_data.tensors[1].clone()
        mini_batched_labeled_data = TensorDataset(t1.repeat(repeated_times,1,1,1), t2.repeat(repeated_times))
    return labeled_data, mini_batched_labeled_data


def get_unlabeled_data():
    return datasets.MNIST('dataset/', train=True, transform=transforms.ToTensor(), download=True)

def get_test_data():
    return datasets.MNIST('dataset/', train=False, transform=transforms.ToTensor(), download=True)