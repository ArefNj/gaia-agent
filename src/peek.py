# src/peek.py
import sys

from src.tools.web import read_webpage

if __name__ == "__main__":
    print(read_webpage(sys.argv[1], sys.argv[2]))