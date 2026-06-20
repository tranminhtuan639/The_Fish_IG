# utils/split_dataset.py
import os, shutil, random

def split(src_dir, dst_dir, train=0.7, val=0.2, test=0.1):
    for label in os.listdir(src_dir):
        images = os.listdir(f"{src_dir}/{label}")
        random.shuffle(images)
        
        n = len(images)
        splits = {
            "train": images[:int(n*train)],
            "val":   images[int(n*train):int(n*(train+val))],
            "test":  images[int(n*(train+val)):]
        }
        
        for split_name, files in splits.items():
            out = f"{dst_dir}/{split_name}/{label}"
            os.makedirs(out, exist_ok=True)
            for f in files:
                shutil.copy(f"{src_dir}/{label}/{f}", f"{out}/{f}")

split("data/raw", "data/processed")