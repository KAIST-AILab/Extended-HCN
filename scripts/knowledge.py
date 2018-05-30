# list of entity types
ENTITY_TYPE_LST = [
    '<R_common>', '<R_atmosphere>', '<R_cuisine>', '<R_location>', '<R_number>', '<R_price>',
    '<R_restrictions>', '<R_rating>', '<R_phone>', '<R_address>', '<R_name>'
]

UNSEEN_SLOT = '<R_restrictions>'

# system response template list
SYS_RES_TEMP_LST = [
    'ok let me look into some options for you',
    'api_call',
    'i\'m on it',
    'hello what can i help you with today',
    'sure is there anything else to update',
    'you\'re welcome',
    'what do you think of this option <R_name>',
    'great let me do the reservation',
    'sure let me find another option for you',
    'here it is <R_address>',
    'here it is <R_phone>',
    'whenever you\'re ready',
    'the option was <R_name>',
    'i am sorry i don\'t have an answer to that question',
    'is there anything i can help you with',
    '<clarify>'
]

# system response template dictionary
# {id: response}
SYS_RES_TEMP_DICT = {k: v for v, k in enumerate(SYS_RES_TEMP_LST)}

BLOCK_LIST = ['ok let me look into some options for you',
              'api_call']


BLOCK_VEC = [1 for i in SYS_RES_TEMP_LST]
for b in BLOCK_LIST:
    idx = SYS_RES_TEMP_DICT[b]
    BLOCK_VEC[idx] = 0

NON_BLOCK_VEC = [1 for i in SYS_RES_TEMP_LST]

# clarify question dictionary
CLR_QST_DICT = {
    '<R_cuisine>': 'any preference on a type of cuisine',
    '<R_location>': 'where should it be',
    '<R_number>': 'how many people would be in your party',
    '<R_price>': 'which price range are you looking for',
    '<R_atmosphere>': 'are you looking for a specific atmosphere',
    '<R_restrictions>': 'do you have any dietary restrictions'
}


# get clarifying order
def get_clr_order(unseen_slot=False):
    if not unseen_slot:
        return ['<R_atmosphere>', '<R_cuisine>', '<R_location>', '<R_number>', '<R_price>']
    return ['<R_atmosphere>', '<R_cuisine>', '<R_location>', '<R_number>', '<R_price>', '<R_restrictions>']


# get api call order
def get_api_order(unseen_slot=False):
    if not unseen_slot:
        return ['<R_cuisine>', '<R_location>', '<R_number>', '<R_price>', '<R_atmosphere>']
    return ['<R_cuisine>', '<R_location>', '<R_number>', '<R_price>', '<R_atmosphere>', '<R_restrictions>']


# get action mask that is determined by the context feature
def get_action_mask(context):
    if context == [1]:
        return NON_BLOCK_VEC
    else:
        return BLOCK_VEC


# determine if the story is terminated
def is_terminate(utterances):
    if 'you\'re welcome' in utterances:
        return True
    return False


# determine if the system finds out that the user have accepted a recommendation
def is_accepted(act_template):
    if 'great let me do the reservation' == act_template:
        return True
    return False


# determine if the system recommends
def is_recommend(act_template):
    if 'what do you think of this option' in act_template:
        return True
    return False


# determine if the system response is containing any placeholder to be filled
def is_contain_placeholder(act_template):
    if '<clarify>' == act_template or 'api_call' in act_template:
        return True
    if '<R_' in act_template:
        return True
    return False
