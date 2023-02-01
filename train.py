from torch.utils.data import Dataset
from mydataloader import dataset

import torch
import torch.nn as nn
import numpy as np
import datetime
import pytz

import numpy as np
import pandas as pd

GENE_EMBED_DIM = 128
GENE_EXPRESSION_VEC = 15077
NUM_CELL_STATES = 5

###############################################################################
# Generator
###############################################################################

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.generator_stack = nn.Sequential(
            nn.Linear(input_dim, 10240),
            nn.LeakyReLU(),
            nn.Linear(10240, 5120),
            nn.BatchNorm1d(5120, momentum=.9),
            nn.LeakyReLU(),
            nn.Linear(5120, 2560),
            nn.BatchNorm1d(2560, momentum=.9),
            nn.LeakyReLU(),
            nn.Linear(2560, 2560),
            nn.BatchNorm1d(4096, momentum=.9),
            nn.LeakyReLU(),
            nn.Linear(2560, 5120),
            nn.BatchNorm1d(5120, momentum=.9),
            nn.LeakyReLU(),
            nn.Linear(5120, 10240),
            nn.BatchNorm1d(10240, momentum=.9),
            nn.LeakyReLU(),
            nn.Linear(10240, output_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x, cond):
        a = torch.cat((x, cond), axis=-1)
        return self.generator_stack(a)

###############################################################################
# Discriminator
###############################################################################

class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Discriminator, self).__init__()
        self.discriminator_stack = nn.Sequential(
            nn.Linear(input_dim, 10240),
            nn.LeakyReLU(),
            nn.Linear(10240, 5120),
            nn.BatchNorm1d(5120, momentum=.9),
            nn.LeakyReLU(),
            nn.Linear(5120, 2560),
            nn.BatchNorm1d(2560, momentum=.9),
            nn.LeakyReLU(),
            nn.Linear(2560, 1280),
            nn.BatchNorm1d(1280, momentum=.9),
            nn.LeakyReLU(),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512, momentum=.9),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, momentum=.9),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64, momentum=.9),
            nn.LeakyReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
	    return self.discriminator_stack(x)

###############################################################################
# Training
###############################################################################

def train():

    # Create Models
    G = Generator(GENE_EXPRESSION_VEC+GENE_EMBED_DIM, GENE_EXPRESSION_VEC).to(device)
    D = Discriminator(GENE_EXPRESSION_VEC, 1).to(device)

    # Define Losses
    lossG = nn.BCELoss()
    lossD = nn.BCELoss()

	alpha = 0.5

    # Create Optimizer
    param_list = list(G.parameters()) + list(D.parameters())

    optimizer_all = torch.optim.Adam(param_list, lr=learning_rate)

    # Train Models
    G.train()
    D.train()

    # training loop
    for ep in range(num_epochs):

        # Reset Loss List
        epoch_loss = {'loss_G': [], 'loss_D': []}

        for index, batch_inputs in enumerate(train_loader):

            # unpack train loader
            KO_gene, batch_X, batch_y = batch_inputs

            # zero all gradients
            optimizer_all.zero_grad()

            # generator input: embedding of knockout gene, unperturbed gene expression
            fake = G(batch_X.to(device), KO_gene)

            # get discriminator decision: real or fake
            decision_fake = D(fake)        
            decision_real = D(batch_Y.to(device))
            
			# Get Losses
            loss_G = lossG(decision_fake, torch.ones_like(decision_fake).to(device))
            loss_D = .5 * lossD(decision_real, torch.ones_like(decision_real).to(device))
            loss_D += .5 * lossD(decision_fake, torch.zeros_like(decision_fake).to(device))

            # Backward the Loss
            loss_total = alpha*loss_G + (1.0 - alpha)*loss_D
            loss_total.backward()

            # Step
            optimizer_all.step()

            # Add Losses to Epoch Dictionary
            epoch_loss["loss_G"].append(loss_G)
            epoch_loss["loss_D"].append(loss_D)

        #Average Loss Every Epoch
        avg_ep_error_G = sum(epoch_loss["loss_G"]) / len(epoch_loss["loss_G"])
        avg_ep_error_D = sum(epoch_loss["loss_D"]) / len(epoch_loss["loss_G"])

        # Print losses every epoch
        print("######################################################")
        print("Generator_Loss: {}\t at epoch: {}".format(avg_ep_error_G, ep))
        print("Discriminator_Loss: {}\t at epoch: {}".format(avg_ep_error_D, ep))

    return G, D


###############################################################################
# Run the Training
###############################################################################

if __name__ == '__main__':
    
	print("Starting...")
	G, D  = train()

	#Save Model
	FILE_MODEL_G = "model/Model_G.pth"
	FILE_MODEL_D = "model/Model_D.pth"
	torch.save(G, FILE_MODEL_G)
	torch.save(D, FILE_MODEL_D)
