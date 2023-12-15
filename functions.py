import numpy as np
import random
import pandas as pd
import math
import re
from statistics import mean

### Makes the data representations more universal across the different products
def generalizeData(data: pd.DataFrame):
    # Title
    data['Title'] = data['Title'].str.lower()
    data['Title'] = data['Title'].apply(lambda x: re.sub('(newegg\.com|thenerds\.net|best buy|diag\.|class)', '', x))
    data['Title'] = data['Title'].apply(lambda x: re.sub('(-inch|inches| inch|\"|in\.|inch| inches| inc\.)', 'inch', x ))
    data['Title'] = data['Title'].apply(lambda x: re.sub('(\.0inch)', 'inch', x ))
    data['Title'] = data['Title'].apply(lambda x: re.sub('(hertz| hz|-hz)', 'hz', x ))
    data['Title'] = data['Title'].apply(lambda x: re.sub('(ledlcd|LED-LCD)', 'led lcd', x ))
    data['Title'] = data['Title'].apply(lambda x: re.sub('( -| /|\(|\)|)', '', x ))
    data['Title'] = data['Title'].apply(lambda x: re.sub('( \. )', ' ', x ))
    data['Title'] = data['Title'].apply(lambda x: re.sub('(refurbished|class|series|diag\.|led|hdtv)', '', x ))
    # Response Time
    data['Response Time'] = data['Response Time'].str.lower()
    data['Response Time'] = data['Response Time'].apply(lambda x: re.sub('( ms)', 'ms', str(x)))
    data['Response Time'] = data['Response Time'].apply(lambda x: re.sub('(ms.*)', 'ms', str(x)))
    # Weight
    data['Weight'] = data['Weight'].str.lower()
    data['Weight'] = data['Weight'].apply(lambda x: re.sub('( lb\.| lbs\.|lb\.|lbs\.|lb| lb| pounds|pounds)', 'lbs', str(x)))
    data['Weight'] = data['Weight'].apply(lambda x: re.sub('(lbs.*)', 'lbs', str(x)))    
    # Width
    data['Width'] = data['Width'].str.lower()
    data['Width'] = data['Width'].apply(lambda x: re.sub('( inches)', 'inches', str(x)))
    data['Width'] = data['Width'].apply(lambda x: re.sub('(lbs.*)', 'lbs', str(x)))
    # Brand
    data['Brand'] = data['Brand'].astype(str)
    # UPC
    data['UPC'] = data['UPC'].astype(str)
    # Product height
    data['Product Height (with stand)'] = data['Product Height (with stand)'].astype(str)
    data['Product Height (without stand)'] = data['Product Height (without stand)'].astype(str)

### Deletes the words from the Title of products that only occur for that product (and hence connot be compared)
def deleteSingleTitleWords(data: pd.DataFrame):
    allWords = []
    
    for feature_i in data["Title"]:
        for word in feature_i.split():
            allWords.append(word)
    
    wordCount = pd.Series(allWords).value_counts()
    wordCount.name = "Count"
    
    wordCount = wordCount.rename_axis('Word').reset_index()
    singleWord = wordCount[wordCount['Count'] == 1]
    
    for i in singleWord:
        data["Title"] = data["Title"].str.replace(r'\b{}\b'.format(i), ' ')
    

### Deletes the feature-values of products that only occur for that product (and hence connot be compared)
def deleteSingleValues(data, feature_list):
    for feature in feature_list:
        
        wordCount = data[feature].value_counts()
        singlevalues = wordCount[wordCount==1].keys()
        
        for i in singlevalues:
            data[feature] = data[feature].apply(lambda x: 'None' if re.match('{}'.format(i), str(x)) else x)

### Splits the string representation of a Title- or feature-value for a product up into a set of separate words
def shingles(feature_str):
    splitted = feature_str.split()
    return(set(splitted))

### Creates the binary vector representation for each product and return a matrix where each column is a feature vector for one product
### This functions also returns the number of rows in the final matrix for each feature, which is used later on in classification
def bin_vec(feature_list, data):
    n = len(data)
    total_feature_vect = np.empty((1,n))
    sections_features = []

    for feature in feature_list:
        shingle_union = set([])
        column =  data[feature]
        for i in column:
            if i!='None':
                shingle_union.update(shingles(i))

        vector_repr = [[0]*len(data) for i in range(len(shingle_union))]
        # One-hot encoding
        shingle_union = list(shingle_union)
        for index in range(n):
            ft_string = data[feature][index]
            if ft_string!='None':
                shingle_i = list(shingles(ft_string))
                match_index = [shingle_union.index(j) if j in shingle_union else -10 for j in shingle_i]
                for j in match_index:
                    if j>=0:
                        vector_repr[j][index] = 1

        # Convert nested list to numpy array for easier use later on
        vector_repr = np.array(vector_repr)

        # Add this to the total matrix
        total_feature_vect = np.append(total_feature_vect, vector_repr, axis=0)
        sections_features.append(len(vector_repr))

    # Delete first 'empty' row
    total_feature_vect = np.delete(total_feature_vect, 0, axis=0)

    return total_feature_vect, sections_features

### Creates the signatures for each product using the binary vector representations 
### and stores them in a matrix where each column contains the signature for one product
def minhashing(feature_vec, h):
    [m,n] = feature_vec.shape    
    # Create an empty signature matrix S of size (h, #products)
    S = np.zeros((h, n), dtype=int)
    
    for i in range(h):
        # Generate a random permutation for each iteration
        permutation = random.sample(range(m), m)
        # Generate the corresponding permutated feature_vector
        feature_vec_perm = feature_vec[permutation,:]
        for j in range(n):
            row_index = np.where(feature_vec_perm[:,j]==1)[0][0]
            # Set the corresponding entry in the signature matrix
            S[i, j] = row_index

    return S

### Implements the locality-sensitive hashing using the signatures of the products. 
### Returns a list of all the found candidate pairs
def LSH(S, r):
    r = int(r)
    candidatePairs = []
    [v,w] = S.shape
    n_bands = math.ceil(v/r)
    BandedSignature = [[0]*w for l in range(n_bands)]
    for i in range(w):
        band_cnt = 0
        # Seperate the signature into bands
        split_indices = [r*l for l in range(1, n_bands)]
        for bandSignature in np.array_split(S[:,i], split_indices):
            my_list = str(bandSignature)
            hashValue = hash(my_list)  
            BandedSignature[band_cnt][i] = hashValue
            band_cnt += 1
        
    for i in range(w-1):
        for k in range(i+1, w):
            for b in range(n_bands):
                if BandedSignature[b][i] == BandedSignature[b][k]:
                    candidatePairs.append([i, k])
                    break
                
    return (candidatePairs)

### Calculates the Cosine Similarity between two binary vectors a and b
def CosineSim(a, b):
    similarity = np.dot(a ,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return similarity

### Using the candidates pairs returned by LSH, is classifies if the pairs are duplicates or not
### Returns a 'updated' list of product pairs classified to be duplicates
def classification(candidatepairs, threshold, data, ft_array, sections):
    candidatepairs2 = []
    weight = 0.2
    sections_idx = np.cumsum(np.array(sections))
    for candidatepair in candidatepairs:
        sim = 0
        c1 = candidatepair[0]
        c2 = candidatepair[1]
        if data["Shop"][c1] != data["Shop"][c2]:
          if (data["Brand"][c1] =='None') or (data["Brand"][c2] =='None') or (data["Brand"][c1] == data["Brand"][c2]):
            if (data["UPC"][c1] == data["UPC"][c2]) and (data["UPC"][c1] !='None') and (data["UPC"][c1] !='None'):
                candidatepairs2.append(candidatepair)
                continue
            title_dist = CosineSim(ft_array[:sections[0],c1],ft_array[:sections[0],c2])
            sim = title_dist
            
            for c in range(len(sections_idx)-1):
                a = ft_array[sections_idx[c]:sections[c+1],c1]
                b = ft_array[sections_idx[c]:sections[c+1],c2]
                if sum(a)!=0 and sum(b)!=0:
                    sim_sec = CosineSim(a,b)
                    sim += sim_sec*weight
            
            if  sim >= threshold:
                candidatepairs2.append(candidatepair)
    return candidatepairs2  

### Determines the number of true duplicate pairs in the dataset
### Returns the number of products in each duplicate cluster and the number of duplicate pairs
def real_duplicates(data):
    modelIDs = []

    # Iterate over rows using iterrows
    for index, row in data.iterrows():
        model_id = row["ModelID"]
        modelIDs.append(model_id)

    modelIDs = pd.DataFrame(modelIDs, columns=["ModelID"])
    countedModelIDList = modelIDs["ModelID"].value_counts()

    duplicateCount = 0               
    for i in countedModelIDList:        
        if i == 2:
            duplicateCount += 1
        elif i == 3:
            duplicateCount += 3
        elif i == 4:
            duplicateCount += 6

    return countedModelIDList, duplicateCount

### Calculates the number of True Positives and False Positives among the list of candidate pairs provided
def estimated_duplicates(candidatepairs, data):
    truePositive = 0
    falsepositive = 0
    for candidatepair in candidatepairs:
        c1 = candidatepair[0]
        c2 = candidatepair[1]
        if data["ModelID"][c1] == data["ModelID"][c2]:
            truePositive += 1
        else:
            falsepositive += 1
            
    return truePositive, falsepositive

### Calculates the f1-score given the found candidate pairs and the dataset
def f1(candidatePiars, data):
    totalDuplicates =  real_duplicates(data)[1]
    TP, FP = estimated_duplicates(candidatePiars, data)
    if TP == 0:
        return 0, 0, 0, 0
    FN = (totalDuplicates - TP)
    precision = TP / (TP+FP)
    recall = TP / (TP + FN)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    
    return f1_score, precision, recall, TP

### Calculates the f1* score given the found candidate pairs and the dataset
def f1star(candidatepairs, data):
    n_real_dupl = real_duplicates(data)[1]
    n_compared = len(candidatepairs)
    n_est_dupl = estimated_duplicates(candidatepairs, data)[0]
  
    if n_compared != 0 and n_est_dupl != 0:
        pair_quality = n_est_dupl/n_compared
        pair_completeness = n_est_dupl/n_real_dupl
        
        f1_star = 2 * ((pair_quality * pair_completeness) / (pair_quality + pair_completeness))
    else:
        pair_quality = 0
        pair_completeness = 0
        f1_star = 0
    
    return pair_quality, pair_completeness, f1_star
   