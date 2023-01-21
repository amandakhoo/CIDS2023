# CIDS2023

Broad Cancer Immunotherapy Data Science Challenge 2023

## To install dependencies

```sh
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If this doesn't work, please let Preston know.
The thing to do will probably be to remove `.venv`,
make your own with `python3 -m venv .venv`
then to follow the steps above.

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
