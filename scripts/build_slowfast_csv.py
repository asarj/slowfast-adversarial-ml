import argparse
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def main(path):
    print(f"Converting {path} to csv...")
    df = pd.read_csv(path, sep=" ", header=None)
    df.columns = ["url", "class"]
    
    df["url"] = "." + df["url"]
    # Save everything
    print(f"Saving the raw csv as test_raw.csv...")
    df.to_csv("./kinetics-400-dataset-files/test_raw.csv", header=False, index=False, sep=" ")
    
    df["url"] = df["url"].replace("/val/", "/val_256/", regex=True)
    print(f"Saving the preprocessed csv as test_256.csv...")
    df.to_csv("./kinetics-400-dataset-files/test_256.csv", header=False, index=False, sep=" ")
    
    print("Done")


if __name__ == '__main__':
    description = 'Helper script for generating the csv that SlowFast uses to train and test'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('-path', "--path", type=str,
                   help=('Path to txt file'))
    main(**vars(p.parse_args()))