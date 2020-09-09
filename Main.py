""" Main function """
import os
import argparse
import Utils
import Config
from Model import *
from Train import emotrain, emoeval
from datetime import datetime
import math
import time
from sklearn.metrics import accuracy_score, confusion_matrix


def main():
    '''Main function'''

    parser = argparse.ArgumentParser()

    # Learning
    parser.add_argument('-lr', type=float, default=2.5e-4)		
    parser.add_argument('-decay', type=float, default=math.pow(0.5, 1/20))	
    parser.add_argument('-epochs', type=int, default=200)		
    parser.add_argument('-patience', type=int, default=10, 
                        help='patience for early stopping')
    parser.add_argument('-save_dir', type=str, default="snapshot", 
                        help='where to save the models')

    # Data
    parser.add_argument('-dataset', type=str, default='IEMOCAP',
                        help='dataset')
    parser.add_argument('-data_path', type=str, required = True,
                        help='data path')
    parser.add_argument('-class_num', type=int, default=4,		
                        help='the hidden size of rnn1')
    # model
    parser.add_argument('-d_raw', type=int, default=33,		
                        help='the hidden size of raw audio feature')
    parser.add_argument('-d_Op', type=int, default=1582,	
                        help='the hidden size of opensmile feature')
    parser.add_argument('-d_h1', type=int, default=300,		
                        help='the hidden size of rnn1')
    parser.add_argument('-d_h2', type=int, default=300,		
                        help='the hidden size of rnn1')
    parser.add_argument('-d_fc', type=int, default=100,		
                        help='the size of fc')
    parser.add_argument('-gpu', type=str, default=None,		
                        help='gpu: default 0')
    parser.add_argument('-report_loss', type=int, default=720,
                        help='how many steps to report loss')
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")

    args = parser.parse_args()
    print(args, '\n')
    # set random seed
    Utils.setup_seed(args.seed)
    
    # Load data field
    print("Loading field...")
    field = Utils.loadFrPickle(args.data_path)
    test_loader = field['test']

    model = Raw_Audio(d_raw = args.d_raw,
                      d_h1 = args.d_h1,
                      d_h2 = args.d_h2,
                      d_fc = args.d_fc,
                      class_num=args.class_num)

    focus_emo = Config.four_iem   
    print("Focused emotion labels {}".format(focus_emo))

    # Train the model
    emotrain(model=model,
                data_loader=field,
                args=args,
                focus_emo=focus_emo)

    # Load the best model to test
    print("Load best models for testing!")
    model = Utils.model_loader(args.save_dir, args.dataset)
    pAccs,y_true,y_pred = emoeval(model=model,
                                  data_loader=test_loader,
                                  args=args,
                                  focus_emo=focus_emo)
    print("Evaluation Metric [{}, {}, {}, {}, {}, {}]".format('happy', 'anger', 'sad', 'neutral', 'WAcc', 'UWAcc'))
    print("Test: ACCs-WA-UWA {}".format(pAccs[:-1]))

    # Save the test results
    record_file = '{}/{}.txt'.format(args.save_dir, args.dataset)
    if os.path.isfile(record_file):
        f_rec = open(record_file, "a")
    else:
        f_rec = open(record_file, "w")
        f_rec.write("Evaluation Metric [{}, {}, {}, {}, {}, {}]\n".format('happy', 'anger', 'sad', 'neutral', 'WAcc', 'UWAcc'))
    f_rec.write("{} - {}\t:\t{}\n".format(datetime.now(), args.lr, pAccs[:-1]))
    f_rec.close()
    
    # Plot the confusion matrix
    print('Plot the confusion matrix...')
    classes = ['hap', 'ang', 'sad', 'neu']
    cm = confusion_matrix(y_true, y_pred)
    Utils.plot_confusion_matrix(cm, 
                              classes, 
                              normalize=False, 
                              figsize=(4, 4),
                              path='./snapshot/IEMOCAP_CM.png')


if __name__ == '__main__':
    main()
