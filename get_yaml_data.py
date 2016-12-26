import yaml
import shelve
import numpy as np
from numpy import array

YAML_FILENAME = 'neha/data/neha_data_revised.yml';


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
    
    
    target_file = open('neha/data/target_list.txt','w');
    target_file.write('\n'.join(target_words));
    target_file.close();
    
    lure_file = open('neha/data/lure_list.txt','w');
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
    neha_data = yaml.load(filestr);
    # compute aggregated results
    aggregated_data = compute_aggregate_results(neha_data);
    # save the reformatted data into a shelve database
    db = shelve.open('neha/data/neha_data.dat','n');
    db['rem_hit'] = rem_hit;
    db['know_hit'] = know_hit;
    db['rem_fa'] = rem_fa;
    db['know_fa'] = know_fa;
    db['CR'] = CR;
    db['miss'] = miss;
    db.close();

def get_word_data(word):
    """
    Computes aggregate classification/confidence data for a given target word.
    
    Args: a string representing the target word.
    
    Returns: a dictionary representing the aggregate data.
    """
    word = word.upper();
    # open yaml file
    ifile = open(YAML_FILENAME,'r');
    # read in data string
    filestr = ifile.read();
    # close file
    ifile.close();
    # parse data string into object (list of dicts)
    all_data = yaml.load(filestr);
    # select only the subset that matches the desired word
    word_trials = [el for el in all_data if el['target']==word];
    # compute aggregated results
    word_data = compute_aggregate_results(word_trials);
    
    return word_data;
    

def compute_aggregate_results(trial_list):
    """
    Computes aggregate classification/confidence data from a list of individual
    trial results.
    
    Args: a list of mappings each representing the results of an individual
    trial.
    
    Returns: a dictionary of classification results each arranged in a format
    similar to that used for the original analysis. I.e., in arrays with two
    columns representing rt and confidence, respectively.
    """
    rem_hit = array([(el['rt.normed'],el['confidence']) for el in trial_list \
                    if el['judgment']=='hit' and el['rk.response']=='remember']);
    know_hit = array([(el['rt.normed'],el['confidence']) for el in trial_list \
                    if el['judgment']=='hit' and el['rk.response']=='know']);
    rem_fa = array([(el['rt.normed'],el['confidence']) for el in trial_list \
                    if el['judgment']=='FA' and el['rk.response']=='remember']);
    know_fa = array([(el['rt.normed'],el['confidence']) for el in trial_list \
                    if el['judgment']=='FA' and el['rk.response']=='know']);
    CR = array([(el['rt.normed'],el['confidence']) for el in trial_list \
                    if el['judgment']=='CR']);
    miss = array([(el['rt.normed'],el['confidence']) for el in trial_list \
                    if el['judgment']=='miss']);
    
    # store these aggregate results in a dictionary
    result = {
        'rem_hit': rem_hit,
        'know_hit': know_hit,
        'rem_fa': rem_fa,
        'know_fa': know_fa,
        'CR': CR,
        'miss': miss
    };
    
    return result;
    