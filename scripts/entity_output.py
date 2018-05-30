import json, util, knowledge
from keras.models import Sequential
from keras.layers import *
from keras import optimizers
from keras import callbacks
from keras.models import Model


class EntityOutput:
    def __init__(self):
        self.next_mention = NextMention()
        self.accept_which = AcceptWhich()
        self.recommend_idx = -1
        self.mention_idx = -1
        self.sorted_rest = None

    def load_weight(self):
        self.next_mention.load_weight()
        self.accept_which.load_weight()

    def train(self):
        self.next_mention.train()
        self.accept_which.train()

    def predict_story(self, api_order, clr_order, sv_pair, sorted_api_res, utterances, act_template):

        if not knowledge.is_contain_placeholder(act_template):
            return act_template

        if act_template == '<clarify>':
            for slot in clr_order:
                if slot not in sv_pair.keys():
                    return knowledge.CLR_QST_DICT[slot]

        if act_template == 'api_call':
            api_values = ['api_call']
            for slot in api_order:
                api_values.append(sv_pair[slot])
            return ' '.join(api_values)

        last_mention_idx, last_recommend_idx, accept_utter = util.get_rest_indice_and_accept_utter(sorted_api_res, utterances)

        if '<R_name>' in act_template:
            # the first recommendation
            if last_recommend_idx == -1:
                mention_which = 'next'

            else:
                request_utter_idx = -1
                while utterances[request_utter_idx] == '<silence>':
                    request_utter_idx -= 2
                mention_which = self.next_mention.predict(utterances[request_utter_idx])

            if mention_which == 'fst':
                mention_idx = 0
            elif mention_which == 'prev':
                mention_idx = last_recommend_idx - 1
            else:
                mention_idx = last_recommend_idx + 1

            rest_name = sorted_api_res[mention_idx]
            return act_template.replace('<R_name>', rest_name)

        if accept_utter is not None:

            accept_which_res = self.accept_which.predict(accept_utter)
            if accept_which_res == 'last_recommended':
                rest_name = sorted_api_res[last_recommend_idx]
            else:
                rest_name = sorted_api_res[last_mention_idx]
            if '<R_phone>' in act_template:
                return act_template.replace('<R_phone>', rest_name+'_phone')
            elif '<R_address>' in act_template:
                return act_template.replace('<R_address>', rest_name+'_address')


class NextMention:
    def __init__(self):
        self.utter_max_words = util.utter_max_words
        self.voc_size = util.voc_size
        self.answer_lst = ['fst', 'prev', 'next']
        self.fst_idx = self.answer_lst.index('fst')
        self.prev_idx = self.answer_lst.index('prev')
        self.next_idx = self.answer_lst.index('next')
        self.answer_size = len(self.answer_lst)
        self.weight_fname = 'nm.h5'
        self.build()

    def build(self):
        input_sent = Input(shape=(self.utter_max_words, self.voc_size), name='input_sent')
        after_lstm = recurrent.LSTM(256)(input_sent)  # (samples, 256)
        action = Dense(3, activation='softmax')(after_lstm)  # (samples, 3)
        self.model = Model(inputs=input_sent, outputs=action)
        optim = optimizers.RMSprop(lr=0.0001)
        self.model.compile(optimizer=optim, loss='binary_crossentropy')
        self.model.summary()

    def load_weight(self):
        self.model.load_weights('weight/'+self.weight_fname)

    def train(self):

        train_x_utter, train_y = self.load_train_data()

        train_x = util.get_multiple_sent_vector(train_x_utter)
        train_y = np.array(train_y)

        stop_callbacks = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', min_delta=0)
        chekpoint = callbacks.ModelCheckpoint('weight/'+self.weight_fname, monitor='val_loss', verbose=1, save_best_only=True,
                                              mode='min')
        self.model.fit(train_x, train_y, batch_size=32, epochs=60, callbacks=[stop_callbacks, chekpoint],
                       validation_split=0.2, shuffle=True)

    def predict(self, sent):
        _, ext_sent = util.extract_sent(sent)
        vector = util.get_multiple_sent_vector([ext_sent])
        prob = self.model.predict(vector)
        ans_idx = np.argmax(prob)
        return self.answer_lst[ans_idx]

    def load_train_data(self):
        print ('Start to load training data for entity output module; next mention')

        fname = util.get_data_fname(task=3, trn=True)
        fd = open(fname, 'r')
        json_data = json.load(fd)
        fd.close()

        i = 0

        train_x = []
        train_y = []

        for story in json_data:
            if i % 100 == 0:
                print(i)
            i += 1

            api_res, utterances = util.split_knowledge(story['utterances'] + [story['answer']['utterance']])
            sorted_rest = util.sort_knowledge(api_res)
            recommend_idx = -1

            for idx, sent in enumerate(utterances):
                y = [0, 0, 0]

                ext_value, _ = util.extract_sent(sent)

                if '<R_name>' in ext_value.keys():

                    # recommendation
                    if knowledge.is_recommend(sent):
                        recommend_idx += 1

                        request_utter_idx = idx - 1
                        if utterances[request_utter_idx] == '<silence>':
                            request_utter_idx -= 2

                        if utterances[request_utter_idx] != '<silence>':
                            _, ext_sent = util.extract_sent(utterances[request_utter_idx])
                            train_x.append(ext_sent)
                            y[self.next_idx] = 1
                            train_y.append(y)

                    # only mention
                    else:
                        _, ext_sent = util.extract_sent(utterances[idx-1])
                        train_x.append(ext_sent)
                        rest_name = sent.split()[-1]
                        rest_idx = sorted_rest.index(rest_name)

                        if rest_idx == 0:
                            y[self.fst_idx] = 1
                        if recommend_idx - rest_idx == 1:
                            y[self.prev_idx] = 1
                        assert y != [0, 0, 0]
                        train_y.append(y)
        return train_x, train_y


class AcceptWhich:
    def __init__(self):
        self.utter_max_words = util.utter_max_words
        self.voc_size = util.voc_size
        self.answer_lst = ['last_recommended', 'last_mentioned']
        self.answer_size = 1
        self.weight_fname = 'aw.h5'
        self.build()

    def build(self):
        self.model = Sequential()
        self.model.add(recurrent.LSTM(256, input_shape=(self.utter_max_words, self.voc_size)))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        self.model.summary()

    def load_weight(self):
        self.model.load_weights('weight/'+self.weight_fname)

    def train(self):

        train_x_utter, train_y = self.load_train_data()

        train_x = util.get_multiple_sent_vector(train_x_utter)
        train_y = np.array(train_y)

        stop_callbacks = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', min_delta=0)
        chekpoint = callbacks.ModelCheckpoint('weight/' + self.weight_fname, monitor='val_loss', verbose=1,
                                              save_best_only=True,
                                              mode='min')
        self.model.fit(train_x, train_y, batch_size=32, epochs=100, callbacks=[stop_callbacks, chekpoint],
                       validation_split=0.2, shuffle=True)

    def predict(self, sent):
        _, ext_sent = util.extract_sent(sent)
        vector = util.get_multiple_sent_vector([ext_sent])
        prob = self.model.predict(vector)[0][0]

        if prob > 0.5:
            return 'last_mentioned'
        else:
            return 'last_recommended'

    def load_train_data(self):
        print('Start to load training data for entity output module; accept which')

        fname = util.get_data_fname(task=5, trn=True)
        fd = open(fname, 'r')
        json_data = json.load(fd)
        fd.close()

        train_x = []
        train_y = []

        i = 0
        for story in json_data:
            if i % 100 == 0:
                print(i)
            i += 1

            _, utterances = util.split_knowledge(story['utterances'] + [story['answer']['utterance']])

            if knowledge.is_terminate(utterances):

                last_mentioned_rest = None
                accept_rest = None
                accept_utter = None
                
                user_turn = True

                for idx, sent in enumerate(utterances):
                    if user_turn:
                        user_turn = False
                    else:
                        ext_values, ext_sent = util.extract_sent(sent)
                        
                        if knowledge.is_accepted(sent):
                            _, accept_utter = util.extract_sent(utterances[idx-1])
    
                        if accept_utter is None:
                            if '<R_name>' in ext_values.keys():
                                last_mentioned_rest = ext_values['<R_name>'][0]
                        else:
                            if '<R_phone>' in ext_values.keys():
                                accept_rest = ext_values['<R_phone>'][0][:-6]
                            elif '<R_address>' in ext_values.keys():
                                accept_rest = ext_values['<R_address>'][0][:-8]
                        user_turn = True

                if accept_rest is not None:
                    train_x.append(accept_utter)
                    if accept_rest == last_mentioned_rest:
                        train_y.append(1)  # last mentioned
                    else:
                        train_y.append(0)  # last recommended

        return train_x, train_y
