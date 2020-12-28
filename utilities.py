import torch
from torch.nn import *
from torch.nn.utils import weight_norm
import torchvision.utils
from torch.nn import MSELoss, CrossEntropyLoss

def weight_normalization(input_dim,output_dim):
    return weight_norm(Linear(input_dim,output_dim))

def cross_entropy_loss(output_real_data, targets):
    loss = CrossEntropyLoss()
    return loss(output_real_data,targets)

def log_sum_exp(x, axis=1):
    max_ = torch.max(x, dim=axis)[0]
    return max_ + torch.log(torch.sum(torch.exp(x - max_.unsqueeze(1)), dim=axis))

def feature_matching(real_features,fake_features):
    mean_squared_error = MSELoss()
    return mean_squared_error(real_features,fake_features)

def tensorboard(writer,batch_number,log_interval, step, unlabeled_loader_1_length,
                supervised_loss, unsupervised_loss, loss, images, Discriminator, Generator, batch_size,writer_real,writer_fake,accuracy):
    if (batch_number + 1) % log_interval == 0:
        print('Training: %d / %d' % (batch_number + 1, unlabeled_loader_1_length))
        step += 1
        with torch.no_grad():
            writer.add_scalars('loss',
                                    {'Supervised Loss': supervised_loss,
                                     'Unsupervised Loss': unsupervised_loss,
                                     'Gen Loss': loss,
                                     'Accuracy': accuracy}, step)

            img_grid_fake = torchvision.utils.make_grid(Generator(batch_size).reshape(-1,1,28,28), normalize=True)
            img_grid_real = torchvision.utils.make_grid(images.reshape(-1,1,28,28), normalize=True)

            writer_fake.add_image(
                "Mnist Fake Images", img_grid_fake, global_step=step
            )
            writer_real.add_image(
                "Mnist Real Images", img_grid_real, global_step=step
            )
            writer.add_histogram('Real Feature',Discriminator(images, feature=True)[0], step)
            writer.add_histogram('Fake Feature',Discriminator(Generator(batch_size),feature=True)[0], step)

            writer.add_histogram('Generator weight normalized layer bias', Generator.weight_normalization_[0].bias, step)
            writer.add_histogram('Discriminator output weight', Discriminator.layers[-1].weight, step)

            Discriminator.train()
            Generator.train()

    return step
