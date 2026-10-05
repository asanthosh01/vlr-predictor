"""Supported training entry point: use the chronological evaluation pipeline."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluate import train

if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",default="data/raw/matches.csv")
    parser.add_argument("--out",default="artifacts")
    args=parser.parse_args()
    train(args.data,args.out)
