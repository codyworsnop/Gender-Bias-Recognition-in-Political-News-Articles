#This file runs pretrain_and_finetune.py

import pretrain_and_finetune as pretrain
pf = pretrain()
'''
Uncomment the option you wish to run; our results were obtained with  option 1
dirty is a bool and refers to whether the news bias dataset has been cleaned or not
notBaseline is a bool and determines if a pretrain on all the news will be done or not
cleenatn is a bool and refers to whether all the news should be cleaned before pretraining; NOTE: this takes A LONG TIME

NOTE: to run this file, articles must have been collected and run_preprocessor.py must have been run
'''

#OPTION 1: run pretrain and fineTune on cleanatn, then on cleaned newsbias dataset
fine_tuned_model = pf.pretrain_and_fineTune(dirty = False, notBaseline=True, cleanatn = True)

#OPTION 2: run pretain and fineTune on dirtyatn, then on cleaned newbias dataset
#fine_tuned_model = pf.pretrain_and_fineTune(dirty = False, notBaseline=True, cleanatn = False)

#OPTION 3: run a baseline comparison on just clean newsbias
#fine_tuned_model = pf.pretrain_and_fineTune(dirty = False, notBaseline=False, cleanatn = False)

#OPTION 4: run a baseline comparison on just dirty newsbias
#fine_tuned_model = pf.pretrain_and_fineTune(dirty = True, notBaseline=False, cleanatn = False)
