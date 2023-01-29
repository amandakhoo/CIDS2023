"""
Utility script to run a jupyter notebook.
Usage: `python3 run_notebook.py <jupyter notebook to run>`
"""

import json
import sys

def main(filepath):
    with open(filepath, 'r') as notebook:
        # Load the notebook: turns out jupyter notebooks are json!
        notebook_content = json.load(notebook)

        # For each cell in the notebook, execute its code.
        for cell in notebook_content['cells']:
            block = "\n".join(cell['source'])
            exec(block)


if __name__ == "__main__":
    main(sys.argv[1])
