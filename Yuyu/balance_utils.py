import pandas as pd
import tensorflow as tf
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class balance_data:
    def __init__(self,directory,mu):
        self.directory = directory
        self.mu = mu
  
    def get_distribution(self):
        label_cnts=[]
        for i in os.listdir(self.directory):
            path=os.path.join(self.directory,i)
            print(path)
            mask = cv2.imread(path,0)
            cnt = np.unique(mask)
            label_cnts.extend(cnt)

        return label_cnts
    
    def get_label_weights(self):
        labels_dict = self.get_distribution(self.directory)
        total = sum(labels_dict.values())
        keys = labels_dict.keys()
        class_weight = dict()

        for key in keys:
            score = np.log(self.mu*total/labels_dict[key])
            print(score)
            class_weight[key] = np.ceil(score) if score > 1.0 else 1.0

        return class_weight



