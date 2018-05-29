import re, json, knowledge
import numpy as np
from keras.preprocessing.sequence import pad_sequences

DATA_PATH = '../data/'

type_dict_fname = 'type_dict.json'
word_dict_fname = 'word_dict.json'

fd = open('type_dict.json', 'r')
type_dict = json.load(fd)
fd.close()

fd = open('word_dict.json', 'r')
word_dict = json.load(fd)
fd.close()

utter_max_words = 30
story_max_utter = 40
voc_size = len(word_dict)


# split sent into words
def split_sent(sent):
    token = re.split('[.,:\s]+', sent.lower())
    return token[:-1] if token[-1] == '' else token


# get entity name of a word
def get_type(word):
    if word in type_dict.keys():
        return type_dict[word]
    return '<R_common>'


# get one-hot encoded vector of a word
def get_word_vector(word):
    vector = np.zeros(voc_size)
    vector[word_dict[word]] = 1
    return vector


# get one-hot encoded vector of a sentence
def get_sent_vector(sent):
    vector = [[get_word_vector(word) for word in sent.split()]]
    pad_vector = pad_sequences(vector, maxlen=utter_max_words, padding='post')
    return pad_vector


# get one-hot encoded vector of multiple sentences
def get_multiple_sent_vector(sent_lst):
    vector = [[get_word_vector(word) for word in sent.split()] for sent in sent_lst]
    pad_vector = pad_sequences(vector, maxlen=utter_max_words, padding='post')
    return pad_vector


# get bag-of-words vector of an utterance
def get_sent_bow_vector(sent):
    vector = np.zeros(voc_size)
    for word in sent.split():
        vector[word_dict[word]] = 1
    return vector


# make 16-dim action vector
def get_action_vector(act_template):
    vector = np.zeros(len(knowledge.SYS_RES_TEMP_LST))

    if act_template is None:
        return vector

    idx = knowledge.SYS_RES_TEMP_DICT[act_template]
    vector[idx] = 1
    return vector


# get action template of a bot utterance
def get_action_template(bot_sent):
    if bot_sent is None:
        return None
    if 'api_call' in bot_sent:
        return 'api_call'
    if bot_sent in knowledge.CLR_QST_DICT.values():
        return '<clarify>'

    _, ext_sent = extract_sent(bot_sent)

    assert ext_sent in knowledge.SYS_RES_TEMP_LST
    return ext_sent


# entity extraction; replace words into its field name
def extract_sent(sent):
    ext_values = {}
    ext_words = []

    for word in sent.split():
        word_type = get_type(word)
        if word_type == '<R_common>':
            ext_words.append(word)
        else:
            ext_words.append(word_type)
            if word_type not in ext_values.keys():
                ext_values[word_type] = []
            ext_values[word_type].append(word)

    ext_sent = ' '.join(ext_words)
    return ext_values, ext_sent


# find utterance indices including 'api_call'
def get_api_sent(utterances):
    api_idx_lst = []
    for idx, sent in enumerate(utterances):
        if 'api_call' in sent:
            api_idx_lst.append(idx)
    return api_idx_lst


# generate slot-value pairs from api call
def get_api_sv(api_order, api_sent):
    assert 'api_call' in api_sent
    slot_value = {}
    for idx, word in enumerate(api_sent.split()[1:]):
        slot = api_order[idx]
        slot_value[slot] = word
    return slot_value


# find last mentioned restaurant index
# find last recommended restaurant index
# find user utterance which decided to accept the recommendation
def get_rest_indice_and_accept_utter(sorted_api_res, utterances):
    last_mention_idx = -1
    last_recommend_idx = -1
    accept_utter = None
    user_turn = True

    for idx, sent in enumerate(utterances):
        if user_turn:
            user_turn = False
        else:
            ext_values, ext_sent = extract_sent(sent)
            if '<R_name>' in ext_sent:
                rest_name = ext_values['<R_name>'][0]
                last_mention_idx = sorted_api_res.index(rest_name)
                if knowledge.is_recommend(ext_sent):
                    last_recommend_idx = sorted_api_res.index(rest_name)

            if knowledge.is_accepted(ext_sent):
                accept_utter = utterances[idx - 1]

            user_turn = True
    return last_mention_idx, last_recommend_idx, accept_utter


# split utterance and knowledge
def split_knowledge(utter_lst):
    api_result = []
    utterances = []
    for sent in utter_lst:
        token = split_sent(sent)

        if get_type(token[0]) == '<R_name>':
            api_result.append(' '.join(token))
        else:
            utterances.append(' '.join(token))

    return api_result, utterances


# sort api result by ratings
def sort_knowledge(api_result):
    rest_dict = {}

    for line in api_result:
        if 'r_rating' in line:
            words = line.split()
            rest_name = words[0]
            rest_rating = int(words[-1])
            rest_dict[rest_name] = rest_rating

    rest_sort = sorted(rest_dict, key=rest_dict.__getitem__, reverse=True)
    return rest_sort


# remove useless punctuation marks
def remove_mark(sent):
    return ' '.join(split_sent(sent))


# get file name of knowledge base
def get_kb_fname(oov=False):
    if not oov:
        return DATA_PATH + 'extendedkb1.txt'
    return DATA_PATH + 'extendedkb2.txt'


# get file name of data
def get_data_fname(task, trn=True, oov=False, unseen_slot=False):
    assert task in range(1, 6)

    if trn:
        return DATA_PATH + 'trn/task%d.json' % task
    else:
        if not oov and not unseen_slot:
            return DATA_PATH + 'tst/tst1/task%d.json' % task
        if oov and not unseen_slot:
            return DATA_PATH + 'tst/tst2/task%d.json' % task
        if not oov and unseen_slot:
            return DATA_PATH + 'tst/tst3/task%d.json' % task
        if oov and unseen_slot:
            return DATA_PATH + 'tst/tst4/task%d.json' % task
