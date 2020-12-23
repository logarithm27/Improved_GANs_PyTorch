import torch
from torch.optim import Adam
from torch import device
from utilities import log_sum_exp,tensorboard
from torch import mean
from torch.nn.functional import softplus
from torch.utils.data import DataLoader
import tensorboardX
from Generator import Generator
from Discriminator import Discriminator
from Data import *


DEVICE = device('cuda')
EPOCHS = 50
BATCH_SIZE = 100
TENSORBOARD_INTERVAL_LOG = 100
EVALUATION_INTERVAL = 1

class SemiSupervisedGan():
    def __init__(self, Discriminator, Generator, labeled_data, unlabeled_data,test_data,tiled_data):
        self.Discriminator = Discriminator
        self.Generator = Generator
        self.Discriminator = self.Discriminator.to(device=DEVICE)
        self.Generator = self.Generator.to(device=DEVICE)
        self.labeled_data = labeled_data
        self.unlabeled_data = unlabeled_data
        self.test_data = test_data
        self.discriminator_optimizer = Adam(self.Discriminator.parameters(),lr=0.003, betas=(0.5, 0.999))
        self.generator_optimizer = Adam(self.Generator.parameters(),lr=0.003,betas=(0.5, 0.999))
        self.mini_batched_labeled_data = tiled_data
        self.writer = tensorboardX.SummaryWriter(log_dir="Tensorboard/")
        self.writer_fake = tensorboardX.SummaryWriter(f"Tensorboard/Fake")
        self.writer_real = tensorboardX.SummaryWriter(f"Tensorboard/Real")

    def train_discriminator(self, labeled_data, unlabeled_data,targets):
        labeled_data, unlabeled_data,targets = labeled_data.to(device=DEVICE), unlabeled_data.to(device=DEVICE), targets.to(device=DEVICE)
        output_labeled_data,output_unlabeled_data= self.Discriminator(labeled_data),self.Discriminator(unlabeled_data)
        fake_output = self.Discriminator(self.Generator(unlabeled_data.size()[0]).view(unlabeled_data.size()).detach())
        log_z_labeled,log_z_unlabeled,log_z_fake = log_sum_exp(output_labeled_data),log_sum_exp(output_unlabeled_data),log_sum_exp(fake_output)
        prob_labeled = torch.gather(output_labeled_data, 1, targets.unsqueeze(1))
        supervised_loss =  - mean(prob_labeled) + mean(log_z_labeled)
        unsupervised_loss = 0.5 * (
                    -mean(log_z_unlabeled) + mean(softplus(log_z_unlabeled)) +  # real_data: log Z/(1+Z)
                    mean(softplus(log_z_fake)))  # fake_data: log 1/(1+Z)
        loss = supervised_loss + unsupervised_loss
        accuracy = mean((output_labeled_data.max(1)[1] == targets).float())
        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()
        return supervised_loss.data.to(device=DEVICE), unsupervised_loss.data.to(device=DEVICE), accuracy

    def train_generator(self, unlabeled_data):
        fake = self.Generator(unlabeled_data.size()[0]).view(unlabeled_data.size())
        mom_gen, fake_output = self.Discriminator(fake, feature=True)
        mom_unlabeled, _ = self.Discriminator(unlabeled_data, feature=True)
        mom_gen = mean(mom_gen, dim=0)
        mom_unlabeled = mean(mom_unlabeled, dim=0)
        loss = mean((mom_gen - mom_unlabeled) ** 2)
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.generator_optimizer.step()
        return loss.data.to(device=DEVICE)

    def train_model(self):
        step = 0
        for epoch in range(EPOCHS):
            self.Discriminator.train()
            self.Generator.train()
            unlabeled_data_1 = DataLoader(self.unlabeled_data, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
            unlabeled_data_2 = DataLoader(self.unlabeled_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True).__iter__()
            labeled_data = DataLoader(self.mini_batched_labeled_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True).__iter__()
            supervised_loss = unsupervised_loss = loss = accuracy = batch_number = 0.
            for unlabeled_images_1,_ in unlabeled_data_1:
                batch_number+=1
                unlabeled_images_2, _ = unlabeled_data_2.next()
                images, digits = labeled_data.next()
                images, unlabeled_images_1, unlabeled_images_2 = images.to(device=DEVICE), unlabeled_images_1.to(device=DEVICE), unlabeled_images_2.to(device=DEVICE)
                digits = digits.type(torch.int64)
                digits = digits.to(device=DEVICE)
                labeled_l, unlabeled_l, acc = self.train_discriminator(images,unlabeled_images_1, digits)
                supervised_loss += labeled_l
                unsupervised_loss += unlabeled_l
                accuracy += acc
                generator_l = self.train_generator(unlabeled_images_2)
                if epoch > 1 and generator_l > 1: # train generator twice
                    generator_l = self.train_generator(unlabeled_images_2)
                loss += generator_l
                step = tensorboard(self.writer,batch_number,TENSORBOARD_INTERVAL_LOG,step,len(unlabeled_data_1),labeled_l,unlabeled_l,generator_l,images,
                            self.Discriminator,self.Generator,BATCH_SIZE, self.writer_real,self.writer_fake,acc) # output logs in tensorboard
            supervised_loss /= batch_number
            unsupervised_loss /= batch_number
            loss /= batch_number
            accuracy /= batch_number
            print(f"Epoch {epoch +1 }, Supervised Loss = {supervised_loss},"
                  f" Unsupervised Loss = {unsupervised_loss}, Loss ={loss} "
                  f"Train accuracy = {accuracy}")
            if (epoch + 1) % EVALUATION_INTERVAL == 0:
                print("Evaluating Data :  %d / %d correct" % (self.eval(), self.test_data.__len__()))

    def predict(self, x):
        with torch.no_grad():
            prediction = torch.max(self.Discriminator(x), 1)[1].data
        return prediction

    def eval(self):
        self.Generator.eval()
        self.Discriminator.eval()
        images, targets = [], []
        for (image, target) in self.test_data:
            images.append(image)
            targets.append(target)
        x, y = torch.stack(images).to(device=DEVICE), torch.LongTensor(targets).to(device=DEVICE)
        pred = self.predict(x)
        return torch.sum(pred == y)

generator = Generator(100)
discriminator = Discriminator()
labeled_data = get_labeled_data(balanced=True)
unlabeled_data = get_unlabeled_data()
test_data = get_test_data()
custom_labeled_data, tiled_data = labeled_data[0], labeled_data[1]
gan = SemiSupervisedGan(discriminator,generator,custom_labeled_data,unlabeled_data,test_data,tiled_data)
gan.train_model()