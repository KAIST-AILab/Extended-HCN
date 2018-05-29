import json
from keras.layers import *
from keras import callbacks
from keras.models import Sequential
import util, knowledge


# Entity tracking module
class EntityTracking:
    def __init__(self):
        self.model = None
        self.utter_max_words = util.utter_max_words
        self.voc_size = util.voc_size
        self.answer_lst = ['fst', 'snd', 'neither']
        self.answer_size = len(self.answer_lst)
        self.neither_idx = self.answer_lst.index('neither')
        self.weight_fname = 'et.h5'
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(recurrent.LSTM(256, input_shape=(self.utter_max_words, self.voc_size)))
        self.model.add(Dense(self.answer_size, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.model.summary()

    def load_weight(self):
        self.model.load_weights('weight/'+self.weight_fname)

    def train(self):

        train_x, train_y = self.load_train_data()

        vector_x = util.get_multiple_sent_vector(train_x)
        vector_y = np.array(train_y)
        stop_callbacks = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', min_delta=0)
        chekpoint = callbacks.ModelCheckpoint('weight/' + self.weight_fname, monitor='val_loss', verbose=1,
                                              save_best_only=True,
                                              mode='min')
        self.model.fit(vector_x, vector_y, batch_size=32, epochs=60, callbacks=[stop_callbacks, chekpoint],
                       validation_split=0.2, shuffle=True)

    # construct training data
    def load_train_data(self):
        print('Start to load training data for entity tracking module')
        train_x = []
        train_y = []

        api_order = knowledge.get_api_order(unseen_slot=False)

        task1_fname = util.get_data_fname(task=1, trn=True)
        task2_fname = util.get_data_fname(task=2, trn=True)

        fd = open(task1_fname, 'r')
        task1_data = json.load(fd)
        fd.close()

        fd = open(task2_fname, 'r')
        task2_data = json.load(fd)
        fd.close()

        i = 0
        for story in task1_data+task2_data:
            if i % 100 == 0:
                print i
            i+=1
            _, utterances = util.split_knowledge(story['utterances']+[story['answer']['utterance']])
            api_idx_lst = util.get_api_sent(utterances)

            # first api call (for task 1, 2)
            if len(api_idx_lst) > 0:
                fst_api_idx = api_idx_lst[0]
                fst_api_sent = utterances[fst_api_idx]
                sent_label_pair = self.get_sent_label_pair(api_order, fst_api_sent, utterances[:fst_api_idx])

                for slot in api_order:
                    for pair in sent_label_pair[slot]:
                        train_x.append(pair[0])
                        train_y.append(pair[1])

            # seconde api call (for task 2)
            if len(api_idx_lst) > 1:
                fst_api_idx = api_idx_lst[0]
                snd_api_idx = api_idx_lst[1]
                snd_api_sent = utterances[snd_api_idx]
                sent_label_pair = self.get_sent_label_pair(api_order, snd_api_sent, utterances[fst_api_idx+1:snd_api_idx])

                for slot in api_order:
                    for pair in sent_label_pair[slot]:
                        train_x.append(pair[0])
                        train_y.append(pair[1])
        return train_x, train_y

    # generate (sentence, label) pair
    def get_sent_label_pair(self, api_order, api_sent, utterances):
        api_sv = util.get_api_sv(api_order, api_sent)
        sent_label_pair = {k: [] for k in api_order}

        for sent in utterances:
            ext_values, ext_sent = util.extract_sent(sent)

            for slot in ext_values.keys():
                value = api_sv[slot]
                slot_sent = ext_sent.replace(slot, '<R_value>')

                if value in ext_values[slot]:
                    val_idx = ext_values[slot].index(value)
                else:
                    val_idx = self.neither_idx
                vect_y = [0, 0, 0]
                vect_y[val_idx] = 1
                sent_label_pair[slot].append([slot_sent, vect_y])

        for slot in api_order:
            for idx in range(len(sent_label_pair[slot])-1):
                sent_label_pair[slot][idx][1] = [0, 0, 1]

        return sent_label_pair

    # predict updated slot-value pair for a given utterance
    def predict(self, sv_pair, ext_values, ext_sent):

        for slot in ext_values.keys():
            replaced_sent = ext_sent.replace(slot, '<R_value>')
            input_vect = util.get_multiple_sent_vector([replaced_sent])

            prob = self.model.predict(input_vect)
            ans_idx = np.argmax(prob)

            if ans_idx != self.neither_idx:
                if len(ext_values[slot])>ans_idx:
                    value = ext_values[slot][ans_idx]
                else:
                    value = ext_values[slot][0]
                sv_pair[slot] = value

        return sv_pair

    # get context feature
    def get_context(self, api_order, sv_pair):
        if '<R_name>' in sv_pair.keys():
            return [1]
        for slot in api_order:
            if slot not in sv_pair.keys():
                return [0]
        return [1]
