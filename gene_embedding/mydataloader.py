from torch.utils.data import Dataset
import torch
import csv
import random
import numpy as np
import pandas as pd

###############################################################################
# Load Data
###############################################################################
class TrainingData(Dataset):

    def __init__(self,
                 path_gene_emb,
                 path_perturb_gene_expr,
                 path_unperturb_gene_expr):

        # gene embedding
        self.gene_emb = pd.read_csv(path_gene_emb)

        # gene KOs 
        self.KOgenes = []

        # knockout gene expression
        self.perturb_seq = pd.read_csv(path_perturb_gene_expr)

        # unperturbed cell gene expression
        self.unperturb_seq = pd.read_csv(path_unperturb_gene_expr)

        # number of genes that were knocked out
        self.n_KOgenes = len(self.KOgenes) 

        # number of unperturbed cells
        self.n_cells_unperturbed = self.unperturbed_seq.shape[0]

    def __getitem__(self, index):

        # select knockout gene at random
        rand_gene_idx = random.randint(0, len(self.n_KOgenes) - 1)
        KOgene = self.KOgenes[rand_gene_idx]

        # get embedding of KO gene
        gene_emb = #TODO

        # get gene expression for a random unperturbed cell
        rand_index = random.randint(0, len(self.n_control) - 1)
        X = np.array(self.control.iloc[rand_index])

        # get gene expression for a random perturbed cell (with selected gene knocked out)
        y = #TODO

        return gene_emb, X, y

    def __len__(self):
        return self.n_KOgenes


dataset = TrainingData("data/gene_embeddings.csv",
                       "data/perturb_gene_expression.csv",
                       "data/unperturbed_gene_expression.csv"
                      )  
print(len(dataset))
