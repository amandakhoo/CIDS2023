FROM python:3.8-slim-buster

WORKDIR /app

# Install dependencies. We copy over the requirements file over
# first since that shouldn't change often, letting us reuse
# previously used Docker layers that have our dependencies.
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "run_notebook.py", "test-notebook.ipynb" ]
