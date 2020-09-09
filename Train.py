"""
Train on Emotion dataset
"""
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import Utils
import math
import Config
from tqdm import trange


def emotrain(model, data_loader, args, focus_emo):
	"""
	:data_loader input the whole field
	"""
	# start time
	time_st = time.time()
	decay_rate = args.decay

	# Load in the training set and validation set
	train_loader = data_loader['train']
	dev_loader = data_loader['dev']

	labels = train_loader['label']
	audio_feats = train_loader['audio']
	raw_audio_feats = train_loader['raw_audio']

	# Optimizer
	lr = args.lr
	model_opt = optim.Adam(model.parameters(), lr=lr)

	print("Dataset : {} \n Emotion rates {}".format(args.dataset,Config.data_count))

	# Raise the .train() flag before training
	model.train()

	over_fitting = 0
	cur_best = -1e10
	cur_best_loss = 100
	glob_steps = 0
	report_loss = 0
	for epoch in range(1, args.epochs + 1):
		model_opt.param_groups[0]['lr'] *= decay_rate	
		labels, audio_feats, raw_audio_feats = Utils.shuffle_lists(labels, audio_feats, raw_audio_feats)
		print("===========Epoch {}==============".format(epoch))
		print("-{}-{}".format(epoch, Utils.timeSince(time_st)))
		for bz in trange(len(labels)):
			# Tensorize a dialogue, a dialogue is a batch

			label = Utils.ToTensor(labels[bz])          
			audio_feat = torch.from_numpy(np.array(audio_feats[bz])).float()
			raw_audio_feat = torch.from_numpy(np.array(raw_audio_feats[bz])).float()
			
			audio_len = raw_audio_feats[bz][:,:,0]
			mask_len = torch.from_numpy(np.array(audio_len)).float()
			audio_lens = Utils.ToAudioLens(audio_len)           
			
					
			label = Variable(label)
			audio_feat = Variable(audio_feat)
			raw_audio_feat = Variable(raw_audio_feat)

			if args.gpu != None:
				os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
				device = torch.device("cuda: 0")
				model.cuda(device)
				label = label.cuda(device)
				audio_feat = audio_feat.cuda(device)
				raw_audio_feat = raw_audio_feat.cuda(device)
				mask_len = mask_len.cuda(device)

			log_prob = model(raw_audio_feat, audio_lens, mask_len)
			loss = comput_class_loss(log_prob, label)
			loss.backward()
			report_loss += loss.item()
			glob_steps += 1

			# gradient clip
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

			model_opt.step()
			model_opt.zero_grad()

			if glob_steps % args.report_loss == 0:
				print("Steps: {} Loss: {} LR: {}".format(glob_steps, report_loss/args.report_loss, model_opt.param_groups[0]['lr']))
				report_loss = 0

		# validate
		pAccs, y_true, y_pred = emoeval(model=model, 
										data_loader=dev_loader,
										args=args, 
										focus_emo=focus_emo)
		print("Evaluation Metric [{}, {}, {}, {}, {}, {}]".format('happy', 'anger', 'sad', 'neutral', 'WAcc', 'UWAcc'))
		print("Validate: ACCs-WA-UWA {}".format(pAccs[:-1]))

		last_best = pAccs[-1]  # loss
		if last_best < cur_best_loss:
			Utils.model_saver(model, args.save_dir, args.dataset)
			cur_best_loss = last_best
			over_fitting = 0
		else:
			over_fitting += 1
		if over_fitting >= args.patience:
			break


def comput_class_loss(log_prob, target):
	""" Weighted loss function """
	loss = F.nll_loss(log_prob, target.view(target.size(0)), reduction='sum')
	loss /= target.size(0)

	return loss


def emoeval(model, data_loader, args, focus_emo):
	""" data_loader only input 'dev' """
	model.eval()

	TP = np.zeros([args.class_num], dtype=np.long) # recall
	TP_FN = np.zeros([args.class_num], dtype=np.long) # gold
	focus_idx = [Config.label_index[emo] for emo in focus_emo]

	audio_feats = data_loader['audio']
	labels = data_loader['label']
	raw_audio_feats = data_loader['raw_audio']

	val_loss = 0
	y_true=[]
	y_pred=[]

	for bz in range(len(labels)):
		label = Utils.ToTensor(labels[bz])       
		audio_feat = torch.from_numpy(np.array(audio_feats[bz])).float()
		raw_audio_feat = torch.from_numpy(np.array(raw_audio_feats[bz])).float()
		
		audio_len = raw_audio_feats[bz][:,:,0]
		mask_len = torch.from_numpy(np.array(audio_len)).float()
		audio_lens = Utils.ToAudioLens(audio_len) 
				
		label = Variable(label)
		audio_feat = Variable(audio_feat)
		raw_audio_feat = Variable(raw_audio_feat)

		if args.gpu != None:
			os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
			device = torch.device("cuda: 0")
			model.cuda(device)
			label = label.cuda(device)
			audio_feat = audio_feat.cuda(device)
			raw_audio_feat = raw_audio_feat.cuda(device)
			mask_len = mask_len.cuda(device)

		log_prob = model(raw_audio_feat, audio_lens, mask_len)

		# print(log_prob, label)
		# val loss
		loss = comput_class_loss(log_prob, label)
		val_loss += loss.item()

		# accuracy
		emo_predidx = torch.argmax(log_prob, dim=1)
		emo_true = label.view(label.size(0))

		a = emo_true.cpu().tolist()
		for i in a:
			y_true.append(i)
		b = emo_predidx.cpu().tolist()
		for j in b:
			y_pred.append(j)

		for lb in range(emo_true.size(0)):
			idx = emo_true[lb].item()
			TP_FN[idx] += 1
			if idx in focus_idx:
				if emo_true[lb] == emo_predidx[lb]:
					TP[idx] += 1

	f_TP = [TP[Config.label_index[w]] for w in focus_emo]
	f_TP_FN = [TP_FN[Config.label_index[w]] for w in focus_emo]
	Recall = [np.round(tp/tp_fn*100, 2) if tp_fn>0 else 0 for tp,tp_fn in zip(f_TP,f_TP_FN)]
	wRecall = sum([r * w / sum(f_TP_FN) for r,w in zip(Recall, f_TP_FN)])
	uRecall = sum(Recall) / len(Recall)

	# Accuracy of each class w.r.t. the focus_emo, the weighted acc, and the unweighted acc
	Total = Recall + [np.round(wRecall,2), np.round(uRecall,2)] + [np.round(val_loss,3)]

	# Return to .train() state after validation
	model.train()

	return Total, y_true, y_pred
