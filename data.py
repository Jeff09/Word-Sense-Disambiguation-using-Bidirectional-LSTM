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

random.seed(0)

train_path2 = './data/senseval2/eng-lex-sample.training.xml'
test_path2 = './data/senseval2/eng-lex-samp.evaluation.xml'
train_path3 = './data/senseval3/EnglishLS.train.mod'
test_path3 = './data/senseval3/EnglishLS.test.mod'

replace_target = re.compile(b"""<head.*?>.*</head>""")
replace_newline = re.compile(b"""\n""")
replace_dot = re.compile(b"\.")
replace_cite = re.compile(b"'")
replace_frac = re.compile(b"[\d]*frac[\d]+")
replace_num = re.compile(b"\s\d+\s")
rm_context_tag = re.compile(b'<.{0,1}context>')
rm_cit_tag = re.compile(b'\[[eb]quo\]')
rm_markup = re.compile(b'\[.+?\]')
rm_misc = re.compile(b"[\[\]\$`()%/,\.:;-]")

# stemmer = porter.PorterStemmer()


def clean_context(ctx_in):
    ctx = replace_target.sub(b' <target> ', ctx_in)
    ctx = replace_newline.sub(b' ', ctx)  # (' <eop> ', ctx)
    ctx = replace_dot.sub(b' ', ctx)     # .sub(' <eos> ', ctx)
    ctx = replace_cite.sub(b' ', ctx)    # .sub(' <cite> ', ctx)
    ctx = replace_frac.sub(b' <frac> ', ctx)
    ctx = replace_num.sub(b' <number> ', ctx)
    ctx = rm_cit_tag.sub(b' ', ctx)
    ctx = rm_context_tag.sub(b'', ctx)
    ctx = rm_markup.sub(b'', ctx)
    ctx = rm_misc.sub(b'', ctx)
    return ctx


def split_context(ctx):
    # word_list = re.split(', | +|\? |! |: |; ', ctx.lower())
    word_list = [word for word in re.split(b', | +|\? |! |: |; ', ctx.lower()) if word]
    return word_list  #[stemmer.stem(word) for word in word_list]


def one_hot_encode(length, target):
    y = np.zeros(length, dtype=np.float32)
    y[target] = 1.
    return y


def load_train_data(se_2_or_3):
    if se_2_or_3 == 2:
        return load_senteval2_data(train_path2, True)
    elif se_2_or_3 == 3:
        return load_senteval3_data(train_path3, True)
    elif se_2_or_3 == 23:
        two = load_senteval2_data(train_path2, True)
        three = load_senteval3_data(train_path3, True)
        return two + three
    else:
        raise ValueError('2, 3 or 23. Provided: %d' % se_2_or_3)

def load_test_data(se_2_or_3):
    if se_2_or_3 == 2:
        return load_senteval2_data(test_path2, False)
    elif se_2_or_3 == 3:
        return load_senteval3_data(test_path3, False)
    elif se_2_or_3 == 23:
        two = load_senteval2_data(test_path2, False)
        three = load_senteval3_data(test_path3, False)
        return two + three
    else:
        raise ValueError('2 or 3. Provided: %d' % se_2_or_3)

def load_senteval3_data(path, is_training):
    return load_senteval2_data(path, is_training, False)

def load_senteval2_data(path, is_training, dtd_validation=True):
    data = []
    parser = et.XMLParser(dtd_validation=dtd_validation)
    doc = et.parse(path, parser)
    instances = doc.findall('.//instance')

    for instance in instances:
        answer = None
        context = None
        for child in instance:
            if child.tag == 'answer':
                senseid = child.get('senseid')
                if senseid == 'P' or senseid == 'U':  # ignore
                    pass
                else:
                    answer = senseid
            elif child.tag == 'context':
                context = et.tostring(child)
            else:
                raise ValueError('unknown child tag to instance')

        # if valid
        if (is_training and answer and context) or (not is_training and context):
            context = clean_context(context)
            x = {
                'id': instance.get('id'),
                'docsrc': instance.get('docsrc'),
                'context': context,
                'target_sense': answer,  # todo support multiple answers?
                'target_word': instance.get('id').split('.')[0],
            }
            data.append(x)

    return data


def get_lexelts(se_2_or_3):
    items = []
    path = train_path2 if se_2_or_3 == 2 else train_path3
    parser = et.XMLParser(dtd_validation=True)
    doc = et.parse(path, parser)
    instances = doc.findall('.//lexelt')

    for instance in instances:
        items.append(instance.get('item'))

    return items


def target_to_lexelt_map(target_words, lexelts):
    # assert len(target_words) == len(lexelts)

    res = {}
    for lexelt in lexelts:
        base = lexelt.split('.')[0]
        res[base] = lexelt

    return res


def build_sense_ids_for_all(data):
    counter = collections.Counter()
    for elem in data:
        counter.update([elem['target_sense']])

    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    senses, _ = list(zip(*count_pairs))
    sense_to_id = dict(zip(senses, range(len(senses))))

    return sense_to_id


def build_sense_ids(data):
    words = set()
    word_to_senses = {}
    for elem in data:
        target_word = elem['target_word']
        target_sense = elem['target_sense']
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


def build_vocab(data):
    """
    :param data: list of dicts containing attribute 'context'
    :return: a dict with words as key and ids as value
    """
    counter = collections.Counter()
    for elem in data:
        counter.update(split_context(elem['context']))

    # remove infrequent words
    min_freq = 1
    filtered = [item for item in counter.items() if item[1]>=min_freq]

    count_pairs = sorted(filtered, key=lambda x: -x[1])
    words, _ = list(zip(*count_pairs))
    words += ('<pad>', '<dropped>')
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id

def sparse_matrix(context, word_to_id):
    y = np.zeros(len(word_to_id), dtype=np.float32)
    for word in context:
        y[word_to_id[word]] = 1.
    yield y

def build_context(data, word_to_id):
    target_sense_to_context = {}
    for elem in data:
        target_sense_id = elem['id']
        context = split_context(elem['context'])
        #context = sparse_matrix(context, word_to_id)
        if target_sense_id not in target_sense_to_context:
            #target_sense_to_context.update({target_sense:context})
            target_sense_to_context[target_sense_id] = []
        target_sense_to_context[target_sense_id].append(context)
    
    return target_sense_to_context

def build_embedding(target_sense_to_context, embedding_matrix, word_num, EMBEDDING_DIM):
    res = {}
    wordvecs = load_glove(EMBEDDING_DIM)
    for target_sense_id, context_matrix in target_sense_to_context.items():
        embedded_sequences = np.zeros(EMBEDDING_DIM)
        n = 0
        for cont in context_matrix:
            for word in cont:
                n += 1
                if isinstance(word, bytes):
                    word = word.decode('utf-8')
                if word in wordvecs:
                    embedded_sequences += wordvecs[word]
                else:
                    embedded_sequences += np.random.normal(0.0, 0.1, EMBEDDING_DIM)                
        res[target_sense_id] = embedded_sequences/n
    return res
        

def convert_to_numeric(data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_target_id, target_sense_to_context_embedding, is_training=True):
    
    n_senses_sorted_by_target_id = [n_senses_from_target_id[target_id] for target_id in range(len(n_senses_from_target_id))]
    starts = (np.cumsum(np.append([0], n_senses_sorted_by_target_id)))[:-1]
    tot_n_senses = sum(n_senses_from_target_id.values())

    def get_tot_id(target_id, sense_id):
        return starts[target_id] + sense_id

    all_data = []
    target_tag_id = word_to_id[b'<target>']
    for instance in data:
        words = split_context(instance['context'])    
        ctx_ints = [word_to_id[word] for word in words if word in word_to_id]
        stop_idx = ctx_ints.index(target_tag_id)
        xf = np.array(ctx_ints[:stop_idx])
        xb = np.array(ctx_ints[stop_idx+1:])[::-1]               
        
        _instance = Instance()        
        instance_id = instance['id']  
        if instance_id not in target_sense_to_context_embedding:
            continue
        sense_embedding = target_sense_to_context_embedding[instance_id]
        target_word = instance['target_word'] 
        target_id = target_word_to_id[target_word]
        
        
        _instance.id = instance_id
        _instance.xf = xf
        _instance.xb = xb 
        _instance.sense_embedding = sense_embedding
        _instance.target_id = target_id
        
        if is_training:                   
            target_sense = instance['target_sense']        
            senses = target_sense_to_id[target_id]
            sense_id = senses[target_sense] if target_sense else -1
            
            _instance.sense_id = sense_id
            _instance.one_hot_labels = one_hot_encode(n_senses_from_target_id[target_id], sense_id)            
            
        # instance.one_hot_labels = one_hot_encode(tot_n_senses, get_tot_id(target_id, sense_id))

        all_data.append(_instance)

    return all_data

def group_by_target(ndata):
    res = {}
    for key, group in groupby(ndata, lambda inst: inst.target_id):
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
        
        if frac == 0:
            l[target_id] = instances[:]
        else:        
            n_r = int(frac * n)
            if min and n_r < min:
                n_r = min
            n_l = n - n_r
    
            l[target_id] = instances[:n_l]
            r[target_id] = instances[-n_r:]

    return l, r if frac > 0 else l

def get_data(_data, n_step_f, n_step_b):
    forward_data, backward_data, target_sense_ids, sense_embeddings = [], [], [], []
    for target_id, data in _data.items():
        for instance in data:
            xf, xb, target_sense_id, sense_embedding = instance.xf, instance.xb, instance.id, instance.sense_embedding
            
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
            sense_embeddings.append(sense_embedding)
    
    return (np.array(forward_data), np.array(backward_data), np.array(target_sense_ids), np.array(sense_embeddings))

def batchify_grouped(gdata, n_step_f, n_step_b, pad_id, n_senses_from_target_id, EMBEDDING_DIM):
    res = {}
    for target_id, instances in gdata.items():
        batch_size = len(instances)
        xfs = np.zeros([batch_size, n_step_f], dtype=np.int32)
        xbs = np.zeros([batch_size, n_step_b], dtype=np.int32)
        xfs.fill(pad_id)
        xbs.fill(pad_id)

        # x forward backward
        for j in range(batch_size):
            n_to_use_f = min(n_step_f, len(instances[j].xf))
            n_to_use_b = min(n_step_b, len(instances[j].xb))
            if n_to_use_f != 0:
                xfs[j, -n_to_use_f:] = instances[j].xf[-n_to_use_f:]
            if n_to_use_b != 0:
                xbs[j, -n_to_use_b:] = instances[j].xb[-n_to_use_b:]

        # labels
        labels = np.zeros([batch_size, n_senses_from_target_id[target_id]], np.float32)
        for j in range(batch_size):
            labels[j, instances[j].sense_id] = 1.
            
        #sense embedding
        sense_embedding = np.zeros([batch_size, EMBEDDING_DIM], np.float32)
        for j in range(batch_size):
            sense_embedding[j, :] = instances[j].sense_embedding

        res[target_id] = (xfs, xbs, labels, sense_embedding)

    return res


class Instance:
    pass


def batch_generator(is_training, batch_size, data, pad_id, n_step_f, n_step_b, pad_last_batch=False, word_drop_rate=None, permute_order=None, drop_id=None):
    data_len = len(data)
    n_batches_float = data_len / float(batch_size)
    n_batches = int(math.ceil(n_batches_float)) if pad_last_batch else int(n_batches_float)

    random.shuffle(data)

    for i in range(n_batches):
        batch = data[i * batch_size:(i+1) * batch_size]

        xfs = np.zeros([batch_size, n_step_f], dtype=np.int32)
        xbs = np.zeros([batch_size, n_step_b], dtype=np.int32)
        xfs.fill(pad_id)
        xbs.fill(pad_id)

        # x forward backward
        for j in range(batch_size):
            if i * batch_size + j < data_len:
                n_to_use_f = min(n_step_f, len(batch[j].xf))
                n_to_use_b = min(n_step_b, len(batch[j].xb))
                if n_to_use_f:
                    xfs[j, -n_to_use_f:] = batch[j].xf[-n_to_use_f:]
                if n_to_use_b:
                    xbs[j, -n_to_use_b:] = batch[j].xb[-n_to_use_b:]
                if is_training and permute_order:
                    if n_to_use_f:
                        xfs[j, -n_to_use_f:] = xfs[j, -n_to_use_f:][np.random.permutation(range(n_to_use_f))]
                    if n_to_use_b:
                        xbs[j, -n_to_use_b:] = xbs[j, -n_to_use_b:][np.random.permutation(range(n_to_use_b))]
                if is_training and word_drop_rate:
                    n_rm_f = max(1, int(word_drop_rate * n_step_f))
                    n_rm_b = max(1, int(word_drop_rate * n_step_b))
                    rm_idx_f = np.random.random_integers(0, n_step_f-1, n_rm_f)
                    rm_idx_b = np.random.random_integers(0, n_step_b-1, n_rm_b)
                    xfs[j, rm_idx_f] = drop_id # pad_id
                    xbs[j, rm_idx_b] = drop_id # pad_id
        # id
        #instance_ids = [inst.id for inst in batch]`
        instance_ids = [inst.id for inst in batch]
        # labels
        target_ids = [inst.target_id for inst in batch]
        sense_ids = [inst.sense_id for inst in batch]

        if len(target_ids) < batch_size:    # padding
            n_pad = batch_size - len(target_ids)
            target_ids += [0] * n_pad
            sense_ids += [0] * n_pad
            instance_ids += [''] * n_pad

        target_ids = np.array(target_ids).astype(np.int32)
        sense_ids = np.array(sense_ids).astype(np.int32)
        # one_hot_labels = np.vstack([inst.one_hot_labels for inst in batch])

        yield (xfs, xbs, target_ids, sense_ids, instance_ids)
        

def write_submission_file(answers):
    pass


if __name__ == '__main__':
    # load data
    data = load_senteval2_data(train_path2, True)
    test_data = load_senteval2_data(test_path2, False)
    
    # build vocab
    word_to_id = build_vocab(data)
    target_word_to_id, target_sense_to_id, words_nums, n_senses_from_word_id = build_sense_ids(data)
    
    #build context vocab of the target sense
    target_sense_to_context = build_context(data, word_to_id)
    #build context embeddings of the target sense
    embedding_matrix = fill_with_gloves(word_to_id, 100)
    target_sense_to_context_embedding = build_embedding(target_sense_to_context, embedding_matrix, len(word_to_id), 100)

    # make numeric
    ndata = convert_to_numeric(data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_word_id, target_sense_to_context_embedding, is_training=True)
    test_ndata = convert_to_numeric(test_data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_word_id, target_sense_to_context_embedding, is_training = False)
    
    
    n_step_f = 40
    n_step_b = 40
    # batch_generator(50, ndata, word_to_id['<pad>'])
    grouped_by_target = group_by_target(ndata)
    train_data, val_data = split_grouped(grouped_by_target, .2, 2)
    #train_data = batchify_grouped(train_data, n_step_f, n_step_b, word_to_id['<pad>'], n_senses_from_word_id, 100)
    #val_data = batchify_grouped(val_data, n_step_f, n_step_b, word_to_id['<pad>'], n_senses_from_word_id, 100)
    
    test_grouped_by_target = group_by_target(test_ndata)
    test_data_ = split_grouped(test_grouped_by_target, 0)

    train_forward_data, train_backward_data, train_target_sense_ids, train_sense_embedding = get_data(train_data, n_step_f, n_step_b)
    test_forward_data, test_backward_data, test_target_sense_ids, test_sense_embedding = get_data(test_data_, n_step_f, n_step_b)
    