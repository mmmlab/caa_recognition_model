# configure for compatibility with Python 3
from __future__ import (absolute_import, division, print_function)
# standard library imports
import shelve
from collections import namedtuple
# scientific library imports
import numpy as np
from numpy import array
# third party imports
import yaml

YAML_FILENAME = 'caa_model/data/neha_data_revised.yml';

# empirical results structure
ERStruct = namedtuple('ERStruct',['know_hit','rem_hit','know_fa','rem_fa',
                                  'CR','miss']);
# reaction time & confidence structure
RTConf = namedtuple('RTConf',['rt','conf','target']);

def save_word_lists():
    """
    Extracts and saves two lists of unique words from trial data: one for
    target words and one for lure words.
    """
    # open yaml file
    ifile = open(YAML_FILENAME,'r');
    # read in data string
    filestr = ifile.read();
    # close file
    ifile.close();
    # parse data string into object (list of dicts)
    neha_data = yaml.load(filestr);
    
    target_words = [el['target'] for el in neha_data if el['judgment']=='hit' \
               or el['judgment']=='miss'];
    
    target_words = np.unique(target_words);
    
    lure_words = [el['target'] for el in neha_data if el['judgment']=='FA' \
               or el['judgment']=='CR'];
    
    lure_words = np.unique(lure_words);
    
    
    target_file = open('caa_model/data/target_list.txt','w');
    target_file.write('\n'.join(target_words));
    target_file.close();
    
    lure_file = open('caa_model/data/lure_list.txt','w');
    lure_file.write('\n'.join(lure_words));
    lure_file.close();
    


def reformat_revised_data():
    """
    Converts individual trial data from yaml format into aggregate format and
    saves the result in a shelve database.
    """
    # open yaml file
    ifile = open(YAML_FILENAME,'r');
    # read in data string
    filestr = ifile.read();
    # close file
    ifile.close();
    # parse data string into object (list of dicts)
    neha_data = yaml.load(filestr,Loader=yaml.CLoader);
    # compute aggregated results
    aggregated_data = compute_aggregate_results(neha_data);
    # save the reformatted data into a shelve database
    db = shelve.open('caa_model/data/neha_data.dat','n');
    db['empirical_results'] = aggregated_data;
    db.close();

def filter_word_data(word,data):
    """
    Computes aggregate classification/confidence data for a given target word.
    
    Args: a string 'word' representing the target word and an ERStruct 'data'
    representing the unfiltered trial data.
    
    Returns: an ERStruct (named tuple) representing the trial data for the
    target word.
    """
    word = word.upper();
    filtered_data = [];
    for category in data:
        total_trials = len(category.rt);
        trials = [(category.rt[i],category.conf[i],category.target[i]) for i in\
            range(total_trials) if category.target[i]==word];
        if(len(trials)>0):
            filtered_category = RTConf(*(array(el) for el in zip(*trials)));
        else:
            filtered_category = None;
        filtered_data.append(filtered_category);
    return ERStruct(*filtered_data);

def compute_aggregate_results(trial_list):
    """
    Computes aggregate classification/confidence data from a list of individual
    trial results.
    
    Args: a list of mappings each representing the results of an individual
    trial.
    
    Returns: a named tuple representing the classification results in a format
    similar to that used for the original analysis. I.e., in arrays with two
    columns representing rt and confidence, respectively.
    """
    # new version that stores the target words as a part of RTConf
        
    rem_hit_list = [];
    know_hit_list = [];
    rem_fa_list = [];
    know_fa_list = [];
    CR_list = [];
    miss_list = [];
    
    for trial in trial_list:
        trial_tuple = (trial['rt.normed'],trial['confidence'],trial['target']);
        if trial['judgment']=='hit' and trial['rk.response']=='remember':
            rem_hit_list.append(trial_tuple);
        elif trial['judgment']=='hit' and trial['rk.response']=='know':
            know_hit_list.append(trial_tuple);
        elif trial['judgment']=='FA' and trial['rk.response']=='remember':
            rem_fa_list.append(trial_tuple);
        elif trial['judgment']=='FA' and trial['rk.response']=='know':
            know_fa_list.append(trial_tuple);
        elif trial['judgment']=='CR':
            CR_list.append(trial_tuple);
        elif trial['judgment']=='miss':
            miss_list.append(trial_tuple);
    
    # convert tuple lists to rt & confidence structures
    rem_hit = RTConf(*(array(el) for el in zip(*rem_hit_list)));
    know_hit = RTConf(*(array(el) for el in zip(*know_hit_list)));
    miss = RTConf(*(array(el) for el in zip(*miss_list)));
    rem_fa = RTConf(*(array(el) for el in zip(*rem_fa_list)));
    know_fa = RTConf(*(array(el) for el in zip(*know_fa_list)));
    CR = RTConf(*(array(el) for el in zip(*CR_list)));
    
    res = ERStruct(know_hit,rem_hit,know_fa,rem_fa,CR,miss);
    
    return res;
    