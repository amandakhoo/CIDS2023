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
                 path_gene_emb_name,
                 path_gene_emb,
                 path_perturb_gene_name,
                 path_perturb_seq_condition,
                 path_perturb_gene_expr,
                 path_unperturb_gene_expr):

        self.excluded_genes = ['Aqr', 'Bach2', 'Bhlhe40', 'Ets1', 'Fosb', 'Mafk', 'Stat3', 'Fzd1', 'P2rx7', 'Tpt1', 'Dvl1', 'Zfp292', 'Sp140']
        
        # gene embedding
        self.gene_emb = pd.read_csv(path_gene_emb)
        
        #gene names for embedding
        self.gene_emb_name = np.load(path_gene_emb_name, allow_pickle = True).item()

        # gene KOs names
        self.KOgenes = pd.read_csv(path_perturb_gene_name)[1::].values.flatten()
        self.KOgenes = [x for x in self.KOgenes if x not in self.excluded_genes]

        # knockout gene expression
        self.perturb_seq = pd.read_csv(path_perturb_gene_expr)
        self.perturb_seq = self.perturb_seq.set_index('index')
        
        #dictionary for cellid to knockout gene
        self.perturb_seq_condition = pd.read_csv(path_perturb_seq_condition)
        self.perturb_seq_condition =  self.perturb_seq_condition.set_index('Unnamed: 0')

        # unperturbed cell gene expression
        self.unperturb_seq = pd.read_csv(path_unperturb_gene_expr)

        # number of genes that were knocked out
        self.n_KOgenes = len(self.KOgenes) 

        # number of unperturbed cells
        self.n_cells_unperturbed = self.unperturb_seq.shape[0]

    def __getitem__(self, index):

        # select knockout gene name at random
        rand_gene_idx = random.randint(0, self.n_KOgenes - 1) 
        KOgene = self.KOgenes[rand_gene_idx] #gene name

        #get embedding of KO gene
        #emb_idx is the index of the gene embeddings for that particular KO gene
        emb_idx = self.gene_emb_name[KOgene.upper()]
        
        #Embeddings of the KO gene:
        gene_emb = np.array(self.gene_emb.iloc[emb_idx])
        
        
        # get gene expression for a random unperturbed cell
        rand_index = random.randint(0, len(self.unperturb_seq) - 1)
        X = np.array(self.unperturb_seq.iloc[rand_index])

        # get gene expression for a random perturbed cell (with selected gene knocked out)
            #KO_idx is the list of indexes where that cell had the gene of interest (KOgene) knocked out
     
        KO_idx = list(self.perturb_seq_condition[self.perturb_seq_condition['condition'] == KOgene].index)
        num_KO_idx = len(KO_idx)
        rand_KO_index = random.randint(0, num_KO_idx - 1)
        KO_expr = self.perturb_seq.loc[KO_idx[rand_KO_index]]
        
        y = np.array(KO_expr)

        return gene_emb, X, y

    def __len__(self):
        
        return self.n_KOgenes


dataset = TrainingData("./gene_embedding/entity2labeldict_TransE_nepochs_128.npy",
                        "./data/gene_embeddings.csv",
                       "./data/knockout_list_conditions.csv",
                       "./data/perturbed_gene_expression_condition.csv",
                       "./data/perturb_gene_expression.csv",
                       "./data/unperturbed_gene_expression.csv"
                      )  
