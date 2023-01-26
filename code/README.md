# `code`

This folder contains the code for analysis and prediction
as well as a deployment guide.

## Programming language

Python, version 3.8 (from https://github.com/amandakhoo/CIDS2023/blob/2e0686d76175e293cd8d309b57e1267a920d711f/Dockerfile#L1)

## Software dependencies

To run this software, make sure that the data is available in the root directory.

You can run this code either locally or through `docker`.
To run locally, ensure that you have `python3` installed,
then run the following commands:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python3 run_notebook.py test-notebook.ipynb
```

To run through `docker`, run the following command:

```sh
docker-compose up
```

### Python dependencies

The dependencies for our Python code are specified in the `requirements.txt` folder.
