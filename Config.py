""" Constants for sequences """

# word pad and unk
PAD = 0
UNK = 1
PAD_WORD = '<pad>'
UNK_WORD = '<unk>'

# focused classes, emotion

four_iem = ['angry', 'happy', 'sad', 'neutral']

# sup dict
data_count = {'angry': 1103, 'happy': 1636, 'neutral': 1708, 'sad': 1084}
label_index = {'angry':1, 'happy':3, 'sad':2, 'neutral':0}