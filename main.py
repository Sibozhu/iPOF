import statistics
import copy
import os
import argparse
from itertools import repeat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from scipy.io import loadmat
import mat73
import time
import random
from pyod.models.knn import KNN
from pyod.models.cof import COF
from pyod.models.abod import ABOD
from pyod.models.sod import SOD
from pyod.models.lof import LOF
from pyod.models.lscp import LSCP
from pyod.models.iforest import IForest
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.combination import aom, moa, average, maximization, median
from pyod.models.so_gaal import SO_GAAL
from pyod.models.mo_gaal import MO_GAAL
from pyod.utils.utility import standardizer
from six.moves import cPickle as pickle 

random.seed(7) 
np.random.seed(7)

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def get_k_nearest_neighbors(inputs):
    X,k,X_scores,len_data = zip(*inputs)
    for i in k:
        k = i
        break

    for i in len_data:
        len_data = i
        break

    x_dict = dict()
    for idx, val in tqdm(enumerate(X),desc="Preparing Data", total=len_data):
        x_dict[idx] = [val]
        temp_list = []
        temp_lof_score = []
        temp_dist_recorder = [] 
        for idx_prime, val_prime in enumerate(X):
            # dist = np.linalg.norm(val - val_prime, ord=2) # L2 Norm distance
            
            dist = cosine_similarity([val],[val_prime])[0][0] # Cosine Similarity distance
            temp_dist_recorder.append(dist)
            temp_list.append((idx_prime,X_scores[idx_prime], val_prime,dist))
        temp_list_prime=copy.deepcopy(temp_list)

        temp_list_prime=sorted(temp_list_prime, key=lambda x: x[3], reverse=True)
        x_dict[idx].append(temp_list_prime[1:k+1])

        neighbor_list = []
        for _, val_neighbor in enumerate(x_dict[idx][1]):  
            neighbor_list.append(val_neighbor[0])
        
        x_dict[idx] += (neighbor_list,)

    return x_dict

def run_epoch(X,x_dict,X_scores):
    X_scores_new=copy.deepcopy(X_scores)
    for idx, _ in enumerate(X):
        temp_lof_score = []
        for _, val_neighbor in enumerate(x_dict[idx][1]):  
            neighbor_index = val_neighbor[0]
            neighbor_list = x_dict[neighbor_index][2]
            if idx in neighbor_list: #directed
                temp_lof_score.append(X_scores[neighbor_index])

        temp_lof_score.append(X_scores[idx]) #including itself

        if temp_lof_score == []:
            X_scores_new[idx] = X_scores[idx]
        else:
            new_score_candidate = statistics.mean(temp_lof_score) #normal average

            ##### Normal Update #####
            X_scores_new[idx] = new_score_candidate
            #################################

    return X_scores_new

def main(k,num_epoches,n_outlier,data_uri,detector,cache_location,random_sample,sample_rate,use_old_cache):

    origin_k = copy.deepcopy(k)

    data_name = data_uri.split('/')[-1].split('.')[0]

    ##### Always toggle random sample on for shuttle dataset #####
    if data_name == 'shuttle':
        random_sample = True


    logfile = open('logs/'+str(data_name)+'_'+str(detector)+'_k='+str(k)+'_log.txt',"a") 


    if data_name == 'http' or data_name == 'KDDCUP99':

        print(f'reading huge dataset ------> {data_uri}')

        raw_data = mat73.loadmat(data_uri)

        data = np.swapaxes(raw_data['X'],0,1)

        len_data = len(data)
        data = data.tolist()

        data_gt = raw_data['y'][0]
        print(data_gt)

        if random_sample:
            randomseed = random.random()
            data = sorted(data, key=lambda k: randomseed)
            data = data[:round(len_data*sample_rate)]

            data_gt = sorted(data_gt, key=lambda k: randomseed)
            data_gt = data_gt[:round(len_data*sample_rate)]

            len_data = len(data)
            print(f'Number of data points after random sample: {len_data}')
    
    elif data_uri.endswith('.mat'):
        print(f'reading .mat dataset ------> {data_uri}')
        ##### read .mat file #####
        raw_data = loadmat(data_uri)
        data = raw_data['X']
        # print(data)
        len_data = len(data)
        data_gt = raw_data['y']

        if random_sample:
            data = data.tolist()
            randomseed = random.random()
            data = sorted(data, key=lambda k: randomseed)
            data = data[:round(len_data*sample_rate)]

            data_gt = sorted(data_gt, key=lambda k: randomseed)
            data_gt = data_gt[:round(len_data*sample_rate)]

            len_data = len(data)
            print(f'Number of data points after random sample: {len_data}')

        ##########################

    else:
        print(f'reading .data dataset ------> {data_uri}')
        ##### read .data file #####
        data = np.loadtxt(data_uri)
        # for i, row in enumerate(data_reader):
            # print(row)
        len_data = len(data)
        print(f'Number of data points: {len_data}')
        if random_sample:
            data = data.tolist()
            data = sorted(data, key=lambda k: random.random())
            #print(data)
            data = data[:round(len_data*sample_rate)]
            data = np.asarray(data)
            len_data = len(data)
            print(f'Number of data points after random sample: {len_data}')

    ###########################

    print(f'Detector: {detector}')
    print(f'Dataset name: {data_name}')
    print(f'K: {k}')
    logfile.write(f'Detector: {detector} \n') 
    logfile.write(f'Dataset name: {data_name} \n') 
    logfile.write(f'K: {k} \n') 

    if detector == 'lof':
        clf = LocalOutlierFactor(n_neighbors=k)
        y_pred = clf.fit_predict(data)
        X_scores = clf.negative_outlier_factor_

    elif detector == 'knn':
        clf = KNN(n_neighbors=k)
        y_pred = clf.fit(data)
        X_scores = clf.decision_scores_

    elif detector == 'cof':
        clf = COF(n_neighbors=k)
        y_pred = clf.fit(data)
        X_scores = clf.decision_scores_

    elif detector == 'abod':
        clf = ABOD(n_neighbors=k)
        y_pred = clf.fit(data)
        X_scores = clf.decision_scores_

    elif detector == 'sod':
        clf = SOD(n_neighbors=k,ref_set=(k//2))
        y_pred = clf.fit(data)
        X_scores = clf.decision_scores_

    elif detector == 'iforest':
        clf = IForest()
        y_pred = clf.fit(data)
        X_scores = clf.decision_scores_
    
    elif detector == 'lscp':
        detector_list = [LOF(n_neighbors=15), LOF(n_neighbors=20),
                        LOF(n_neighbors=25), LOF(n_neighbors=35)]
        clf = LSCP(detector_list, random_state=42)
        y_pred = clf.fit(data)
        X_scores = clf.decision_scores_

    elif detector == 'featurebagging':
        clf = FeatureBagging(check_estimator=False)
        y_pred = clf.fit(data)
        X_scores = clf.decision_scores_

    elif detector == 'mo_gaal':
        clf = MO_GAAL()
        y_pred = clf.fit(data)
        X_scores = clf.decision_scores_

    elif detector == 'so_gaal':
        clf = SO_GAAL()
        y_pred = clf.fit(data)
        X_scores = clf.decision_scores_

    elif detector == 'max' or detector == 'median' or detector == 'aom' or detector == 'moa':
        n_clf = 20
        # Initialize 20 base detectors for combination
        k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
                150, 160, 170, 180, 190, 200]

        if random_sample:
            X_scores_array = np.zeros([len(data), n_clf])
        else:
            X_scores_array = np.zeros([data.shape[0], n_clf])

        data = standardizer(data)

        print('Combining {n_clf} LOF detectors'.format(n_clf=n_clf))

        for i in range(n_clf):
            k = k_list[i]

            clf = LOF(n_neighbors=k)

            clf.fit(data)
            X_scores_array[:, i] = clf.decision_scores_

        if detector == 'max':
            # Combination by max
            X_scores = maximization(X_scores_array)

        elif detector == 'median':
            # Combination by median
            X_scores = median(X_scores_array)

        elif detector == 'aom':
            # Combination by aom
            X_scores = aom(X_scores_array)

        elif detector == 'moa':
            # Combination by moa
            X_scores = moa(X_scores_array)
    
    else:
        raise Exception('unrecognized detector "{}"'.format(detector))

    y_lim = None
    x_lim = None
    k = origin_k
    if not data_uri.endswith('.mat'):
        fig, ax = plt.subplots()
        #plt.ylim((0.0, 150.0))
        #plt.xlim((0.0, 50.0))

        for idx,_ in enumerate(tqdm(data, desc='Plotting...')):
            if y_pred[idx] == -1:
                plt.plot(data[idx][::2], data[idx][1::2], color='red')
            else:
                plt.plot(data[idx][::2], data[idx][1::2], color='blue')
        
        y_lim = ax.get_ylim()
        x_lim = ax.get_xlim()
        fig.savefig(f'{data_name}_{detector}_k={k}_original.png')

    if data_uri.endswith('.mat'):

        initial_auc = roc_auc_score(data_gt,abs(X_scores))

        print('Initial AUC Score: {0:.4f}'.format(initial_auc))
        logfile.write('Initial AUC Score: {0:.4f} \n'.format(initial_auc))

    if use_old_cache:
        cache_folder = './cache/'
        cache_path = os.path.join(cache_folder, f'{data_name}_{detector}_k={k}_original.pkl')

        if os.path.exists(cache_path):
            print(f"Caches exist: {cache_path}!")
            x_dict = load_dict(cache_path)
        else:
            print("Caches do not exist!")
            ##### extremely time consuming #####
            data_input = zip(data, repeat(k), X_scores, repeat(len_data))
            x_dict = get_k_nearest_neighbors(data_input)
            ####################################
            print("Saving cache...")
            os.makedirs(cache_folder, exist_ok=True)
            print(cache_path)
            save_dict(x_dict, cache_path)

    auc_recorder = []
    running_time_recorder = []

    progress=tqdm(range(num_epoches))
    for _ in progress:
        start_ipof = time.time()
        X_scores = run_epoch(data,x_dict,X_scores)
        ipof_time = time.time() - start_ipof
        running_time_recorder.append(ipof_time)
        if data_uri.endswith('.mat'):
            temp_auc = roc_auc_score(data_gt,abs(X_scores))
            # print(f'Initial AUC Score: {0:.4f}'.format(initial_auc))
            progress.set_description("AUC score: {0:.4f}".format(temp_auc))
            auc_recorder.append(temp_auc)
    logfile.write('Final AUC Score: {0:.4f} \n'.format(temp_auc))
    average_ipof_time = np.mean(running_time_recorder)
    print('Average iPOF time per iteration: {0:.4f} \n'.format(average_ipof_time))
    logfile.write('Average iPOF time per iteration: {0:.4f} \n'.format(average_ipof_time))

    if data_uri.endswith('.mat'):
        fig = plt.figure()
        ax = plt.subplot(111)
        axes = plt.gca()
        ax.set_title(f"{data_name} Dataset Observation, K={k}")
        ax.plot([x for x in auc_recorder])
        plt.ylabel('AUC score')
        plt.xlabel('Iteration')
        plt.ylim((0.0, 1.0))
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        fig.savefig(f'{data_name}_{detector}_k={k}.png')
    
    else:
        raise Exception('unrecognized dataset "{}"'.format(data_uri))
    
    logfile.close()
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    def add_bool_arg(name, default, help):
        arg_group = parser.add_mutually_exclusive_group(required=False)
        arg_group.add_argument('--' + name, dest=name, action='store_true', help=help)
        arg_group.add_argument('--no_' + name, dest=name, action='store_false', help=("Do not " + help))
        parser.set_defaults(**{name:default})    
    parser.add_argument("--k", type=int, help="k nearest neighbors", default=10)
    parser.add_argument("--num_epoches", type=int, help="number of ensemble iterations", default=2000)
    parser.add_argument("--n_outlier", type=int, help="number of outliers", default=6)
    parser.add_argument("--data_uri", type=str, help="path to dataset", default="datasets/breastw.mat")
    parser.add_argument("--detector", type=str, help="Choice of outlier detector", default="lof")
    ##TO-DO: store processed cache
    parser.add_argument("--cache_location", type=str, help="path to cache storage", default="./cache/")
    add_bool_arg('random_sample', default=False, help="whether to random sample 0.01 percent data from the dataset")
    parser.add_argument("--sample_rate", type=float, help="the rate of random sample if you choose to do so", default=0.1)
    add_bool_arg('use_old_cache', default=True, help="whether to look for stored cache of nearest neighbors")
    ##### mass experiments #####    
    arg = parser.parse_args()
    
    main(
        k=arg.k,
        num_epoches=arg.num_epoches,
        n_outlier=arg.n_outlier,
        data_uri=arg.data_uri,
        detector=arg.detector,
        cache_location=arg.cache_location,
        random_sample=arg.random_sample,
        sample_rate=arg.sample_rate,
        use_old_cache=arg.use_old_cache
        )
