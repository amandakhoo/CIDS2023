# CIDS2023

Broad Cancer Immunotherapy Data Science Challenge 2023

## To install dependencies

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If this doesn't work, please let Preston know.

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


#### (S)am's notes if you (like him) are having a lot of trouble with Git remote permissions:
- Set up to track the remote using SSH
- Set up private/public key pair if not already set up
- Make sure the private key is being used by the ssh-agent, a la: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux#adding-your-ssh-key-to-the-ssh-agent

