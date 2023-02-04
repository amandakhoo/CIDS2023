# CIDS2023

Broad Cancer Immunotherapy Data Science Challenge 2023

## To install dependencies

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If this doesn't work, please let us know.

## Downloading data dependencies

Please download our saved models and data from
https://drive.google.com/drive/folders/1eJ0uKNQSzOix2-2oIISwoiWNwfLEWxDs.
Inside there should be saved models from saving y'all the trouble of re-training
and the data that we used for this challenge.

## Quick tour

`classifier.ipynb` trains an svm for classification and saves the trained model. A pre-trained model is available in the google drivel link.

`GAN_SVM_connector.ipynb` runs the main logic for challenge 2. It requires
data and saved models from other steps and outputs csv files for the various
parts of challenge 2.

`GAN.ipynb` trains the GAN we use to generate cell samples.

`mydataloader.py` loads the data and gene embeddings into a format that our GAN can understand.

`run_notebook.py` is a utility file for running jupyter notebooks from the command line.

## To run with Docker

Make sure that docker is installed! Recent versions of
docker should have `docker-compose` bundled with it.
If your docker doesn't have `docker-compose`, install docker-compose.
Run

```sh
docker-compose up
```

with the training data in the directory from which you run the command.
(Right now we just have test infrastructure in place
that emulates outputting the `.csv` file. Try running
`docker-compose up` now for a friendly greeting!)
