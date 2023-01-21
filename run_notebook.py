import json
import sys

def main(filepath):
    with open(filepath, 'r') as notebook:
        notebook_content = json.load(notebook)

        for cell in notebook_content['cells']:
            eval(cell['source'])


if __name__ == "__main__":
    main(sys.argv[0])
