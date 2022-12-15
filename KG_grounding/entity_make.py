import csv
from tqdm import tqdm
import pandas as pd
from itertools import permutations
from multiprocessing import Pool
import enchant
import random
import json
import pickle

# d = enchant.Dict("en_US")
# import spacy

# nlp = spacy.load('en_core_web_sm')
import argparse
import os

parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--dataset_dir", default=".", type=str,
                    help="The dataset dictionary")
parser.add_argument("--save_dataset_dir", default=".", type=str,
                    help="The save processed data to saving dataset dictionary")
parser.add_argument("--OpenKE_dir", default=".", type = str,
                    help="The dictionary to save the processed data for OpenKE")
parser.add_argument("--save_conceptnet",
                    default="../dataset/ko/ko-atomic-trans_reorder.csv",
                    type=str,
                    help="The output conceptnet data dir.")
parser.add_argument("--commongend",
                    default="../dataset/ko/kommongen_dev.src_alpha.txt",
                    type=str,
                    help="The default dir of the commongen dataset.")

args = parser.parse_args()

conceptnet = os.path.join(args.save_conceptnet)
commongend = os.path.join(args.commongend)
# os.makedirs(args.save_dataset_dir, exist_ok = True)
# os.makedirs(args.OpenKE_dir, exist_ok = True)

if 1:
    entity_time = []
    all_entity = []
    for file_name in [commongend, commongend.replace("dev", 'train'), commongend.replace("dev", 'test')]:
        train_entity = []
        dev_entity = []
        test_entity = []
        with open(file_name, encoding='utf-8') as f_concept:
            concept_list = f_concept.readlines()
            for clist in concept_list:
                entitys = clist.split()
                for entity in entitys:
                    if entity not in all_entity:
                        all_entity.append(entity)
                    if "train" in file_name:
                        if entity not in train_entity:
                            train_entity.append(entity)
                    if "dev" in file_name:
                        if entity not in dev_entity:
                            dev_entity.append(entity)
                    if "test" in file_name:
                        if entity not in test_entity:
                            test_entity.append(entity)
                            
                
                
                
with open('entity2id2.txt', 'w') as f:
    for entity in all_entity:
        f.write('entity')