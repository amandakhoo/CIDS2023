# %%
import torch
import torch.nn as nn
import numpy as np
import datetime
import pytz
import pickle
import random

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd

torch.set_default_dtype(torch.float64)

# %%
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
            nn.BatchNorm1d(2560, momentum=.9),
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
        #print(a.size()[0])
        return self.generator_stack(a)

# %%
GENE_EMBED_DIM = 150
GENE_EXPRESSION_VEC = 15077
BATCH_SIZE = 10

input_dim = GENE_EXPRESSION_VEC+GENE_EMBED_DIM
output_dim = GENE_EXPRESSION_VEC

# %%
#def get_data(train_loader):

    #for index, batch_inputs in enumerate(train_loader):
     #   KO_gene, batch_X, batch_y = batch_inputs

    #return KO_gene, batch_X, batch_y


#train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
#KO_gene, batch_X, batch_y = get_data(train_loader)


# %%
def dataloader_mini(KO_gene, gene_emb, gene_emb_label, unperturbed_expr_df, n_cells):

    #get embedding of KO gene
    #emb_idx is the index of the gene embeddings for that particular KO gene
    emb_idx = gene_emb_label[KO_gene.upper()]  #an index
    #Embeddings of the KO gene:
    KO_gene_emb = np.array(gene_emb.iloc[emb_idx]) #an embedding

    #select n random unperturbed cells

    rand_index = random.sample(range(0, len(unperturbed_expr_df) - 1), n_cells)
    unpert_input = np.array(unperturbed_expr_df.iloc[rand_index])

    return KO_gene, unpert_input, KO_gene_emb

# %%
def generate_data(unpert_input, KO_gene_emb, model):
    model.eval()
    z = unpert_input
    cond = KO_gene_emb
    generated_data = model(z, cond)
    return generated_data

# %%
def classify_state_SVM(generated_data):

    #get the gene names for the 15077 genes
    columns = pd.read_csv('./data/unperturbed_gene_expression.csv', nrows = 0).columns
    #get gene names for what the SVM expects
    SVM_filt = pd.read_csv('./data/unperturbed_filtered.csv', nrows = 0).columns
    #turn the generated transcripts into a dataframe for filtering in the next step
    cell = pd.DataFrame(generated_data.detach().numpy(), columns = columns)
    #grab just the genes that the SVM uses to predict t cell state
    SVM_input = cell[SVM_filt]

    #load the svm
    loaded_model = pickle.load(open('./saved_models/svc_model_unperturbed.sav', 'rb'))
    preds = loaded_model.predict(SVM_input)

    return preds

# %%
gene_list = pd.read_csv('./data/unperturbed_gene_expression.csv', nrows = 0).columns.values
gene_emb = pd.read_csv('./data/gene_embeddings.csv')
gene_emb_label = np.load('./gene_embedding/entity2labeldict_TransE_nepochs_128.npy', allow_pickle = True).item()
unperturbed_expr_df = pd.read_csv('./data/unperturbed_gene_expression.csv')

# %%
n_cells = 100

# %%
KO_gene_list = []
KO_gene_list_found = []
KO_gene_list_not_found = []
pred_list = []

model = torch.load("./saved_models/Model_G_300.pth")

for gene in gene_list:

    #load the data and KO gene
    pred = None
    KO_gene_list.append(gene)
    if gene.upper() in gene_emb_label:
        KO_gene, unpert_input, KO_gene_emb = dataloader_mini(KO_gene = gene, gene_emb = gene_emb, gene_emb_label = gene_emb_label, unperturbed_expr_df = unperturbed_expr_df, n_cells = n_cells)
        #unpert_input_list.append(unpert_input)
        #KO_gene_emb_list.append(KO_gene_emb)

        #generate data using GAN
        #batch size comes first
        KO_gene_tensor = torch.tensor(np.tile(KO_gene_emb, (n_cells, 1)))
        #print(unpert_input.shape)
        generated_data = generate_data(torch.tensor(unpert_input), KO_gene_tensor, model)

        #run the svm classifier
        pred = classify_state_SVM(generated_data)
    else:
        KO_gene_list_not_found.append(gene)
        continue
    KO_gene_list_found.append(gene)
    pred_list.append(pred)



# %%
KO_gene_list_found #TODO: how to combine genes not found and genes found in embeddings.

# %%
KO_gene_list_not_found

# %%
def value_or_zero(data, state):
    if state in data:
        return data[state]
    else:
        return 0.0

def tcell_state_proportions(data):
    cell_states = ['progenitor', 'effector', 'terminal exhausted', 'cycling', 'other']
    proportions = []
    for gene in data.index:
        value_counts = data.loc[gene].value_counts()
        total = value_counts.sum()
        state_proportions = value_counts/total

        gene_proportions = {}
        for state in cell_states:
            gene_proportions[state] = value_or_zero(state_proportions, state)
        proportions.append(gene_proportions)
    return pd.DataFrame(proportions, index = data.index)


# %%
# handle genes that aren't in our embedding
for not_found in KO_gene_list_not_found:
    pred_list.append(['other'] * n_cells)

output = pd.DataFrame(pred_list, index = (KO_gene_list_found + KO_gene_list_not_found))

for gene in output.index:
    value_counts = output.loc[gene].value_counts()
    total = value_counts.sum()
    if total == 0:
        total = 1

proportions = tcell_state_proportions(output )
proportions.to_csv("part_c_output.csv", header=False)

# %%
def enough_cycling(data):
  enough_cycling = []
  for gene in data.index:
    if value_or_zero(data.loc[gene], 'cycling') >= 0.05:
      enough_cycling.append(1)
    else:
      enough_cycling.append(0)

  return enough_cycling

def l1_loss_part_a(row):
    return abs(value_or_zero(row, 'cycling') - 0) + \
      abs(value_or_zero(row, 'terminal exhausted') - 0) + \
        abs(value_or_zero(row, 'effector') - 0) + \
          abs(value_or_zero(row, 'other') - 0) + \
            abs(value_or_zero(row, 'progenitor') - 1)

def loss_part_b(row):
  return (value_or_zero(row, 'progenitor') / 0.0675) + \
    (value_or_zero(row, 'effector') / 0.2097) - \
      (value_or_zero(row, 'terminal exhausted') / 0.3134) + \
        (value_or_zero(row, 'cycling') / 0.3921)

# %%
proportions['enough_cycling'] = enough_cycling(proportions)

# %%
proportions.sort_index(key=lambda gene: l1_loss_part_a(proportions.loc[gene]))[['progenitor', 'enough_cycling']].to_csv("part_a_output.csv", header=False)

# %%
proportions['objective_b'] = [loss_part_b(proportions.loc[gene]) for gene in proportions.index]
proportions.sort_index(ascending=False, key=lambda gene: loss_part_b(proportions.loc[gene]))[['objective_b', 'enough_cycling']].to_csv("part_b_output.csv", header=False)


