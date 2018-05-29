import argparse
import numpy as np
import util, knowledge, json
from entity_tracking import EntityTracking
from action_selector import ActionSelector
from entity_output import EntityOutput


class Model:
    def __init__(self):
        self.entity_tracking = EntityTracking()
        self.action_selector = ActionSelector(self.entity_tracking)
        self.entity_output = EntityOutput()

    def load_weight(self, task):
        self.entity_tracking.load_weight()
        self.action_selector.load_weight(task)
        self.entity_output.load_weight()

    def predict_story(self, api_order, clr_order, story):

        api_result, utterances = util.split_knowledge(story['utterances'])
        sorted_api_result = util.sort_knowledge(api_result)

        story_user_utter = []
        story_context = []
        story_bot_utter = []

        user_turn = True
        bot_sent = None

        sv_pair = {}
        context = [0]

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

        prob = self.action_selector.predict_story(story_user_utter, story_context, story_bot_utter)
        action_mask = knowledge.get_action_mask(context)
        masked_prob = np.multiply(prob, action_mask)
        idx = np.argmax(masked_prob)

        act_template = knowledge.SYS_RES_TEMP_LST[idx]

        return self.entity_output.predict_story(api_order, clr_order, sv_pair, sorted_api_result, utterances, act_template)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ts", "--task", choices=[1, 2, 3, 4, 5], type=int)
    parser.add_argument("-t", "--train", action='store_true')
    parser.add_argument("-et", "--entity_track", action='store_true')
    parser.add_argument("-as", "--action_selector", action='store_true')
    parser.add_argument("-eo", "--entity_output", action='store_true')
    parser.add_argument("-us", "--unseen_slot", action='store_true')
    parser.add_argument("-oov", "--oov", action='store_true')

    args = parser.parse_args()
    task = args.task
    oov = args.oov
    unseen_slot = args.unseen_slot

    np.random.seed(2)

    print args

    if args.train:
        if args.entity_track:
            et = EntityTracking()
            et.train()
        if args.action_selector:
            et = EntityTracking()
            et.load_weight()
            acs = ActionSelector(et)
            acs.train(task)
        if args.entity_output:
            eo = EntityOutput()
            eo.train()
    else:
        model = Model()
        model.load_weight(task)
        print task

        fname = util.get_data_fname(task, trn=False, unseen_slot=unseen_slot, oov=oov)
        fd = open(fname, 'r')
        json_data = json.load(fd)
        fd.close()

        api_order = knowledge.get_api_order(unseen_slot=unseen_slot)
        clr_order = knowledge.get_clr_order(unseen_slot=unseen_slot)

        correct_cnt = 0
        for story in json_data:
            if correct_cnt % 100 == 0:
                print correct_cnt
            pred = model.predict_story(api_order, clr_order, story)
            answer = util.remove_mark(story['answer']['utterance'])
            if pred == answer:
                correct_cnt += 1
            else:
                for u in story['utterances']:
                    print u
                print 'pred: ', pred
                print 'answer: ', answer
        print 'task %d: correct %d / %d' % (task, correct_cnt, len(json_data))


if __name__ == '__main__':
    main()
