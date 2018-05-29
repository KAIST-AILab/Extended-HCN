import json, util, knowledge
from keras.models import Sequential
from keras.layers import *
from keras.preprocessing.sequence import pad_sequences
from keras import callbacks


class ActionSelector:
    def __init__(self, et):
        self.entity_tracking = et
        self.voc_size = util.voc_size
        self.utter_max_words = util.utter_max_words
        self.story_max_utter = util.story_max_utter
        self.answer_size = len(knowledge.SYS_RES_TEMP_LST)
        self.context_size = 1
        self.build()

    def build(self):
        self.model = Sequential()
        self.model.add(recurrent.LSTM(128, recurrent_dropout=0.7,
                                      input_shape=(
                                      self.story_max_utter, self.voc_size + self.answer_size + self.context_size)))
        self.model.add(Dense(self.answer_size, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.model.summary()

    def load_weight(self, task):
        fname = 'as_task%d.h5' % task
        self.model.load_weights('weight/'+fname)

    def predict_story(self, story_user_utter, story_context, story_bot_utter):
        story_x = []
        for turn_idx in range(len(story_user_utter)):
            user_bow_vector = util.get_sent_bow_vector(story_user_utter[turn_idx])
            context = story_context[turn_idx]
            bot_vector = util.get_action_vector(story_bot_utter[turn_idx])

            vector_x = np.concatenate((user_bow_vector, context))
            vector_x = np.concatenate((vector_x, bot_vector))
            story_x.append(vector_x)
        train_x = pad_sequences([story_x], maxlen=self.story_max_utter, padding='post')
        train_x = np.array(train_x)
        prob = self.model.predict(train_x)[0]
        return prob

    def train(self, task):
        fname = 'as_task%d.h5' % task
        train_user_utter, train_context, train_bot_utter, train_y_utter = self.load_train_data(task)

        train_x = []

        for story_idx in range(len(train_user_utter)):
            story_x = []
            for turn_idx in range(len(train_user_utter[story_idx])):
                user_bow_vector = util.get_sent_bow_vector(train_user_utter[story_idx][turn_idx])
                context = train_context[story_idx][turn_idx]
                bot_vector = util.get_action_vector(train_bot_utter[story_idx][turn_idx])

                vector_x = np.concatenate((user_bow_vector, context))
                vector_x = np.concatenate((vector_x, bot_vector))
                story_x.append(vector_x)
            train_x.append(story_x)
        train_x = pad_sequences(train_x, maxlen=self.story_max_utter, padding='post')
        train_x = np.array(train_x)

        train_y = []
        for act_template in train_y_utter:
            train_y.append(util.get_action_vector(act_template))

        train_y = np.array(train_y)

        stop_callbacks = callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min', min_delta=0)
        chekpoint = callbacks.ModelCheckpoint('weight/' + fname, monitor='val_loss', verbose=1,
                                              save_best_only=True,
                                              mode='min')
        self.model.fit(train_x, train_y, batch_size=32, epochs=60, callbacks=[stop_callbacks, chekpoint],
                       validation_split=0.2, shuffle=True)

    def load_train_data(self, task):
        print ('Start to load training data for action selector module')

        train_user_utter = []
        train_context = []
        train_bot_utter = []

        train_y = []

        fname = util.get_data_fname(task)
        fd = open(fname, 'r')
        json_data = json.load(fd)
        fd.close()

        api_order = knowledge.get_api_order(unseen_slot=False)

        i = 0

        for story in json_data:

            if i % 100 == 0:
                print(i)
            i += 1

            _, utterances = util.split_knowledge(story['utterances'] + [story['answer']['utterance']])

            story_user_utter = []
            story_context = []
            story_bot_utter = []

            user_turn = True
            bot_sent = None

            sv_pair = {}

            for sent in utterances:
                if user_turn:
                    ext_values, ext_sent = util.extract_sent(sent)

                    sv_pair = self.entity_tracking.predict(sv_pair, ext_values, ext_sent)
                    context = self.entity_tracking.get_context(api_order, sv_pair)

                    story_user_utter.append(ext_sent)
                    story_context.append(context)
                    story_bot_utter.append(bot_sent)

                    user_turn = False

                else:
                    bot_sent = util.get_action_template(sent)
                    user_turn = True

            train_user_utter.append(story_user_utter)
            train_context.append(story_context)
            train_bot_utter.append(story_bot_utter)

            train_y.append(bot_sent)

        return train_user_utter, train_context, train_bot_utter, train_y

