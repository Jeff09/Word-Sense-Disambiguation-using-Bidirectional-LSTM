# -*- coding: utf-8 -*-

import lxml.etree as et
import math
import numpy as np
import collections
import re
import nltk.stem.porter as porter
from nltk.stem.wordnet import WordNetLemmatizer
from itertools import groupby
import random
from glove import *
import os
from glob import glob
from nltk.corpus import wordnet as wn
import csv

random.seed(0)

semcor_path = './google_data/semcor'
masc_path = './google_data/masc'
algorithmic_map = './google_data/algorithmic_map.txt'
manual_map = './google_data/manual_map.txt'

sense_embedding_file = 'sense_embedding.csv'

input_filemask = 'xml'
total_words_wordnet = 206941 #  the total number of definitions in the dictionary WordNet 3.0
EMBEDDING_DIM = 100


rm_markup = re.compile(b'\[.+?\]')
rm_misc = re.compile(b"[\[\]\$`()%/,\.:;-]")
replace_num = re.compile(b"\s\d+\s")
regex = re.compile('[^a-zA-Z]')

def clean_context(ctx_in):
    ctx = replace_num.sub(b' <number> ', ctx_in)
    ctx = rm_markup.sub(b'', ctx)
    ctx = rm_misc.sub(b'', ctx)
    return ctx

def load_masc_file(masc_path):
    list_of_files = []
    for root, dirpath, files in os.walk(masc_path):
        for filename in [f for f in files if f.endswith('.xml')]:
            list_of_files.append(os.path.join(root, filename)) 
    return list_of_files        

def load_semcor_file(semcor_path):    
    list_of_files = glob('%s/*.%s'%(semcor_path, input_filemask))
    return list_of_files

def load_data(list_of_files, dtd_validation=True):
    """
    :param list_of_files: list of filepath
    :return data: list of dictionary containing all words
            word_freq: dictionary containing all word as key, its frequency as value
            target_sense_freq: dictionary containing target_sense as key, its frequency as value
    """
    data = []
    word_freq = {}
    target_sense_freq = {}
    parser = et.XMLParser(dtd_validation=dtd_validation)
    for file_name in list_of_files:    
        doc = et.parse(file_name, parser)
        instance = doc.findall('.//word')
        for child in instance:
            word = None
            sense = None
            if child.tag == "word":
                word = child.get("text").lower()
                if word not in word_freq:
                    word_freq[word] = 0
                word_freq[word] += 1
                x = {
                        "word": word,
                        "break_level": child.get("break_level")
                    }
                sense = child.get("sense")
                if sense:
                    x["lemma"] = child.get("lemma")
                    x["pos"] = child.get("pos")                
                    x["sense"] = sense
                    x["is_target"] = True
                    if sense not in target_sense_freq:
                        target_sense_freq[sense] = 0
                    target_sense_freq[sense] += 1
                else:
                    x["is_target"] = False
                data.append(x)
    return data, word_freq, target_sense_freq

def build_context(data):
    """
    return 
        sense_to_contex: dictionary of all target senses - all context words in one paragraph 
    """
    sense_to_context = {}
    context = []
    word_list = []
    for elem in data:
        word = elem["word"]
        word = regex.sub('', word)
        if elem["break_level"] == "NO_BREAK" and (word == "quot" or not word):
            continue
        if elem["break_level"] != "PARAGRAPH_BREAK":
            context.append(elem["word"])
            if elem["is_target"]:
                word_list.append((elem["word"], elem["sense"]))
        else:
            sense_to_context[tuple(word_list)] = context
            context = [elem["word"]]
            if elem["is_target"]:
                word_list = [(elem["word"], elem["sense"])]
            else:
                word_list = []
    return sense_to_context
            

def NOAD_to_wordnet(data):
    """
    transform target sense annotated with NOAD (New Oxford American Dictionary) word senses into WordNet senses
    """
    NOAD_to_wordnet = {}
    with open(algorithmic_map, 'r') as f:
        lines = f.readlines()
        for line in lines:
            noad, wordnet = line.split()
            NOAD_to_wordnet[noad] = wordnet
    with open(manual_map, 'r') as f:
        lines = f.readlines()
        for line in lines:
            noad, wordnet = line.split()
            NOAD_to_wordnet[noad] = wordnet
    
    count = 0
    for elem in data:        
        if elem["is_target"]:
            if elem["sense"] not in NOAD_to_wordnet:
                count += 1
                continue
            noad_sense = elem["sense"]
            elem["sense"] = NOAD_to_wordnet[noad_sense]
    print("NOAD sense not in mapping text: %d" %count)
    return data

def build_vocab(word_freq):
    """
    :param word_freq: list of dicts containing word - frequency pairs
    :return: a dict with words as key and ids as value
    """
    
    min_freq = 1
    filtered = [item for item in word_freq.items() if item[1]>=min_freq]

    count_pairs = sorted(filtered, key=lambda x: -x[1])
    words, _ = list(zip(*count_pairs))
    #words += ('<pad>', '<dropped>')
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id

def build_sense_ids(data):
    """
    :return 
        target_word_to_id   dictionary of word - word_frequency
        target_sense_to_id  list of dictionary including sense - sense_id 
        len(words):  total number of dictinct words
        n_senses_from_word_id:  dictionary of word_id - number of senses of that word  
    """
    words = set()
    word_to_senses = {}
    for elem in data:
        if elem["is_target"]:
            target_word = elem['word']
            target_sense = elem['sense']
            if target_word not in words:
                words.add(target_word)
                word_to_senses.update({target_word: [target_sense]})
            else:
                if target_sense not in word_to_senses[target_word]:
                    word_to_senses[target_word].append(target_sense)
    
    words = list(words)
    target_word_to_id = dict(zip(words, range(len(words))))
    target_sense_to_id = [dict(zip(word_to_senses[word], range(len(word_to_senses[word])))) for word in words]

    n_senses_from_word_id = dict([(target_word_to_id[word], len(word_to_senses[word])) for word in words])
    return target_word_to_id, target_sense_to_id, len(words), n_senses_from_word_id

def split_context(ctx):
    # word_list = re.split(', | +|\? |! |: |; ', ctx.lower())
    word_list = [word for word in re.split(', | +|\? |! |: |; |\) |\( |\' ', ctx.lower()) if word]
    return word_list

def get_all_wordnet_definition():
    all_definition = []
    for ss in wn.all_synsets():
        ss_list = split_context(ss.definition())
        all_definition.append(ss_list)
    return all_definition

all_definition = get_all_wordnet_definition()

def build_word_occurrence_definition(word):
    count = 0
    for ss in all_definition:
        if word in ss:
            count += 1
    #print(word, count)
    return count

def build_sense_vector(sense, word_freq, wordvecs):
    sense_definition = sense.definition()
    #print("sense_definition"+str(sense_definition))
    #sense_definition_words = sense_definition.split()
    sense_definition_words = split_context(sense_definition)
    sense_vector = np.zeros(EMBEDDING_DIM)
    n = 0
    for word in sense_definition_words:
        word_occurrence_definition = build_word_occurrence_definition(word)
        idf_word = np.log(total_words_wordnet / float(word_occurrence_definition))
        if word in wordvecs:
            sense_vector += wordvecs[word] * idf_word
        else:
            sense_vector += np.random.normal(0.0, 0.1, EMBEDDING_DIM) * idf_word
        n += 1
    return sense_vector/n

def sc2ss(sensekey):
    '''Look up a synset given the information from SemCor'''
    ### Assuming it is the same WN version (e.g. 3.0)
    try:
        return wn.lemma_from_key(sensekey).synset()
    except:
        pass

def build_sense_embedding(target_sense_to_id, word_freq, EMBEDDING_DIM):
    """
    build sense vector for every target sense using the definition of wordnet and glove embedding
    return
        res: dictionary of sense - sense_vector
    """
    res = {}
    wordvecs = load_glove(EMBEDDING_DIM)
    
    for target_sense_list in target_sense_to_id:
        for key, _ in target_sense_list.items():
            sense_vector = np.zeros(EMBEDDING_DIM)
            senses = key.split(',')
            n = 0
            for sensekey in senses:
                #print(sensekey)                
                if '/' in sensekey:
                    continue
                sense_synset = sc2ss(sensekey)
                if sense_synset:
                    sense_vector += build_sense_vector(sense_synset, word_freq, wordvecs)
                    n += 1
            if n != 0:
                res[key] = sense_vector/n
    return res

def get_embedding(sense_embedding_file):
    sense_embeddings_ = None
    with open(sense_embedding_file, 'r', newline='') as f:
        reader = csv.reader(f)
        sense_embeddings_ = dict(reader)
    sense_embeddings__ = {}
    for key, value in sense_embeddings_.items():
        value = value.split()
        vec = np.zeros(len(value)-1)
        for i in range(len(value)):
            if '[' in value[i]:
                continue
                value[i] = value[i][1:]
            elif ']' in value[i]:
                value[i] = value[i][:-1]
            if value[i]:
                vec[i-1] = float(value[i])
        sense_embeddings__[key] = vec    
    return sense_embeddings__

def convert_to_numeric(data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_word_id, sense_to_context, sense_embeddings):
    
    n_senses_sorted_by_target_id = [n_senses_from_word_id[target_id] for target_id in range(len(n_senses_from_word_id))]
    tot_n_senses = sum(n_senses_from_word_id.values())
    print("total senses:", tot_n_senses)

    all_data = []      
    n = 0
    for sense_tuple, context in sense_to_context.items():
        if len(sense_tuple) != 0:
            ctx_ints = [word_to_id[word] for word in context if word in word_to_id]
            for target_word, target_sense in list(sense_tuple):            
                target_id = target_word_to_id[target_word]
                stop_idx = ctx_ints.index(target_id)
                xf = np.array(ctx_ints[:stop_idx])
                xb = np.array(ctx_ints[stop_idx+1:])[::-1]  
                
                _instance = Instance()    
                _instance.target_word = target_word
                _instance.target_id = target_id
                _instance.xf = xf
                _instance.xb = xb            
                
                #print(target_sense)
                try:
                    _instance.target_sense = sense_embeddings[target_sense]
                except:
                    n += 1
                    pass
                all_data.append(_instance)      
    
    print("total number of target sense having not target sense vector %d" % n)
    return all_data

def group_by_target(ndata):
    res = {}
    for key, group in groupby(ndata, lambda inst: inst.target_word):
       res.update({key: list(group)})
    return res

def split_grouped(data, frac, min=None):
    assert frac >= 0.
    assert frac < .5
    l = {}
    r = {}
    for target_id, instances in data.items():
        # instances = [inst for inst in instances]
        random.shuffle(instances)   # optional
        n = len(instances)
        print(n)
        if frac == 0:
            l[target_id] = instances[:]
        else:        
            n_r = int(frac * n)
            if min and n_r < min:
                n_r = min
            n_l = n - n_r
            print(n_l, n_r)
            l[target_id] = instances[:n_l]
            r[target_id] = instances[-n_r:]

    return l, r if frac > 0 else l

def get_data(_data, n_step_f, n_step_b):
    forward_data, backward_data, target_sense_ids, sense_embeddings = [], [], [], []
    for target_id, data in _data.items():
        for instance in data:
            xf, xb, target_sense_id, target_sense = instance.xf, instance.xb, instance.target_id, instance.target_sense
            
            n_to_use_f = min(n_step_f, len(xf))
            n_to_use_b = min(n_step_b, len(xb))
            xfs = np.zeros([n_step_f], dtype=np.int32)
            xbs = np.zeros([n_step_b], dtype=np.int32)            
            if n_to_use_f != 0:
                xfs[-n_to_use_f:] = xf[-n_to_use_f:]
            if n_to_use_b != 0:
                xbs[-n_to_use_b:] = xb[-n_to_use_b:]
            
            forward_data.append(xfs)
            backward_data.append(xbs)
            target_sense_ids.append(target_sense_id)
            sense_embeddings.append(target_sense)
    
    return (np.array(forward_data), np.array(backward_data), np.array(target_sense_ids), np.array(sense_embeddings))
    
class Instance:
    pass

if __name__ == '__main__':

    #masc_files = load_masc_file(masc_path)
    semcor_files = load_semcor_file(semcor_path)
                    
    #masc_data = load_data(masc_files)
    semcor_data, semcor_word_freq, semcor_target_sense_freq = load_data(semcor_files)    
    semcor_data = NOAD_to_wordnet(semcor_data)
    
    sense_to_context = build_context(semcor_data)    
    
    word_to_id = build_vocab(semcor_word_freq)
    target_word_to_id, target_sense_to_id, target_words_nums, n_senses_from_word_id = build_sense_ids(semcor_data)
    
    '''
    sense_embeddings = build_sense_embedding(target_sense_to_id, semcor_word_freq, EMBEDDING_DIM)    
    with open('sense_embedding.csv','w', newline='') as f:
        w = csv.writer(f)
        for key, value in sense_embeddings.items():
            w.writerow([key, value])'''
        
    sense_embeddings_ = get_embedding(sense_embedding_file)  
        
    all_data = convert_to_numeric(semcor_data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_word_id, sense_to_context, sense_embeddings_)
    
    n_step_f = 40
    n_step_b = 40
    grouped_by_target = group_by_target(all_data)
    train_data, val_data = split_grouped(grouped_by_target, .2)
    
    train_forward_data, train_backward_data, train_target_sense_ids, train_sense_embedding = get_data(train_data, n_step_f, n_step_b)
