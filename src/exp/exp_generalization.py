# Script for held-out families test.
from src.evaluate import run as evaluate
from src.train import run

if __name__ == '__main__':
    run()
    evaluate()
