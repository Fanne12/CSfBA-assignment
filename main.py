import numpy as np
import random
import json
import pandas as pd
import math
import re
from statistics import mean
import functions

### Import the data
with open('TVs-all-merged.json', 'r') as json_file:
    data = json.load(json_file)

### Extracts the data on: shop, url, modelID, features_map, title from the jsonfile and creates a dataframe
shop_vector = []
url_vector = []
modelID_vector = []
features_map_vector = []
title_vector = []

for model_id, entries in data.items():
    for entry in entries:
        # Split the values into separate variables
        shop_value = entry['shop']
        shop_vector.append(shop_value)
        
        url_value = entry['url']
        url_vector.append(url_value)
    
        modelID_value = entry['modelID']
        modelID_vector.append(modelID_value)
    
        features_map_value = entry['featuresMap']
        features_map_vector.append(features_map_value)
        
        title_value = entry['title']
        title_vector.append(title_value)
        
data_dict = {
    'Shop': shop_vector,
    'URL': url_vector,
    'ModelID': modelID_vector,
    'Features Map': features_map_vector,
    'Title': title_vector}

# Create a DataFrame from the dictionary
pd_data = pd.DataFrame(data_dict)

### Extracts additional data from the 'featuresMap' and adds it to the dataframe
pd_data["Brand"] = pd_data["Features Map"].apply(lambda x: x.get("Brand") if isinstance(x, dict) else np.nan)
pd_data["UPC"] = pd_data["Features Map"].apply(lambda x: x.get("UPC") if isinstance(x, dict) else np.nan)
pd_data["Response Time"] = pd_data["Features Map"].apply(lambda x: x.get("Response Time") if isinstance(x, dict) else np.nan)
pd_data["Weight"] = pd_data["Features Map"].apply(lambda x: x.get("Weight") if isinstance(x, dict) else np.nan)
pd_data["Product Height (with stand)"] = pd_data["Features Map"].apply(lambda x: x.get("Product Height (with stand)") if isinstance(x, dict) else np.nan)
pd_data["Product Height (without stand)"] = pd_data["Features Map"].apply(lambda x: x.get("Product Height (without stand)") if isinstance(x, dict) else np.nan)
pd_data["Width"] = pd_data["Features Map"].apply(lambda x: x.get("Width") if isinstance(x, dict) else np.nan)

### Specify the list of features that will be used in the product representation
features = ["Title", "Response Time", "Weight" , "Product Height (with stand)", 
            "Product Height (without stand)", "Width"]

### Make the data representations more general across different products
functions.generalizeData(pd_data)  
### Delete words from the Title and features that do not add much for comparison
functions.deleteSingleTitleWords(pd_data)
functions.deleteSingleValues(pd_data, features[1:])

### --------- Bootstrapping -------------------------------------------------
I = 5 # number of bootstraps
p = 0.63 # training/test split

H = 500 # number of minhashes

## Select the number of rows per band to have the threshold t run from 0.05 to 0.95
t = np.arange(0.05, 1, 0.05)
R = np.arange(1, H/10, 1)
B = [int(H/i) for i in R] 
BR = [(1/b)**(1/r) for b,r in zip(B,R)]
R_select = []
for x in t:
    BR_t = [abs(z-x) for z in BR]
    R_select.append(R[BR_t.index(min(BR_t))])

## Grid for the threshold parameter in the classification step
param_t = np.arange(0.6, 0.9, 0.05) 

## Lists to store the optimal values of the threshold and the evaluation measures
opt_t = [[0]*I for k in range(len(R_select))]
metrics_g = [[0]*len(R_select) for k in range(6)] # in order: pair quality, pair completeness, f1*, f1, precision, recall
fraction_of_comparisons_g = [0 for k in range(len(R_select))]

max_f1 = 0
i = 0
while i < I:
    ## Split data randomly in train and test data
    rnd_data = pd_data.sample(frac=1)
    N = len(rnd_data)
    train_size = round(N*p)
    train_data = rnd_data[:train_size]
    test_data = rnd_data[train_size:]
    train_data.reset_index(inplace=True)
    test_data.reset_index(inplace=True)
    if functions.real_duplicates(train_data)[1]/len(train_data) < 0.16:
        i += 1

        for index in range(len(R_select)):
            for T in param_t:            
                ## Run the algorithm for different thresholds T
                [feature_matrix,s] = functions.bin_vec(features, train_data)
                signature_matrix = functions.minhashing(feature_matrix, H)
                cp = functions.LSH(signature_matrix, R_select[index])
                cp2 = functions.classification(cp, T, train_data, signature_matrix, s)
                F1_score = functions.f1(cp2, train_data)[0]
                F1_star_score = functions.f1star(cp, train_data)
            
                if F1_score > max_f1: 
                    T_opt = T
                    max_f1 = F1_score

                print(i, index, T) # keep track of progress while running code
                
            [feature_matrix,s] = functions.bin_vec(features, test_data)
            signature_matrix = functions.minhashing(feature_matrix, H)
            cp = functions.LSH(signature_matrix, R_select[index])
            cp2 = functions.classification(cp, T_opt, test_data, signature_matrix, s)
            F1_score = functions.f1(cp2, test_data)
            F1_star_score = functions.f1star(cp, test_data)
            total_comparisons = len(test_data)*(len(test_data)-1)/2
            fraction_of_comparisons_g[index] += (len(cp)/total_comparisons)
            
            # Store optimal value of threshold T
            opt_t[index][i-1] = T_opt    

            # Store the scores of the evaluation metrics
            metrics_g[0][index] += F1_star_score[0]
            metrics_g[1][index] += F1_star_score[1]
            metrics_g[2][index] += F1_star_score[2]
            metrics_g[3][index] += F1_score[0]
            metrics_g[4][index] += F1_score[1]
            metrics_g[5][index] += F1_score[2]

## Compute the average scores over the bootstraps
metrics_final_g = [[x/I for x in l] for l in metrics_g]
mean_n_comparisons_made_g = [a/I for a in fraction_of_comparisons_g]


