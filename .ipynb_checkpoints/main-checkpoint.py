import pickle
import os
import pandas as pd
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from src.anomalyscores import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
# from beepy import beep
import numpy as np
from src.spot import SPOT
from src.constants import *
from sklearn.metrics import *
import os


	
def convert_to_windows(data, model):
	windows = []; w_size = model.n_window
	
	for i, g in enumerate(data): 
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w if 'TranAD' in args.model or 'Attention' in args.model else w.view(-1))
	return torch.stack(windows)

def load_dataset(dataset):
	global testlabels
	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')
	loader = []
	for file in ['train', 'test', 'labels','testlabels']:
		if dataset == 'SMD': file = 'machine-1-1_' + file
		if dataset == 'SMAP': file = 'P-1_' + file
		if dataset == 'MSL': file = 'C-1_' + file
		if dataset == 'UCR': file = '136_' + file
		if dataset == 'NAB': file = 'ec2_request_latency_system_failure_' + file
		loader.append(np.load(os.path.join(folder, f'{file}.npy'), allow_pickle=True, mmap_mode='r'))

	# loader = [i[:, debug:debug+1] for i in loader]
	if args.less: loader[0] = cut_array(0.2, loader[0])
	train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
	test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
	labels = loader[2]
	testlabels=loader[3].T
	print(test_loader)
	print(test_loader)
	print('testlabels@@@@@@@@@@@@',testlabels)
	return train_loader, test_loader, labels

def save_model(model, optimizer, scheduler, epoch, accuracy_list):
	folder = f'checkpoints/{args.model}_{args.dataset}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model.ckpt'
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(modelname, dims):
	import src.models
	model_class = getattr(src.models, modelname)
	model = model_class(dims).double()

	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
	if os.path.exists(fname) and (not args.retrain or args.test):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(fname)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; accuracy_list = []
	return model, optimizer, scheduler, epoch, accuracy_list

def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):
	l = nn.MSELoss(reduction = 'mean' if training else 'none')
	feats = dataO.shape[1] 
	if 'TranAD' in model.name:
		l = nn.MSELoss(reduction = 'none')
		data_x = torch.DoubleTensor(data); dataset = TensorDataset(data_x, data_x)
		bs = model.batch if training else len(data)
		dataloader = DataLoader(dataset, batch_size = bs)
		n = epoch + 1; w_size = model.n_window
		l1s, l2s = [], []
		if training:
			for d, _ in dataloader:
				#print('TRAIN 1')	
				local_bs = d.shape[0]
				#print('TRAIN 2')	
				window = d.permute(1, 0, 2)
				#print('TRAIN 3')	
				elem = window[-1, :, :].view(1, local_bs, feats)
				#print('TRAIN 4')	
				z = model(window, elem)
				#print('TRAIN 5')	
				l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
				#print('TRAIN 6')	
				if isinstance(z, tuple): z = z[1]
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward(retain_graph=True)
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			for d, _ in dataloader:
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, bs, feats)
				z = model(window, elem)
				if isinstance(z, tuple): z = z[1]
			loss = l(z, elem)[0]
			print('LOSS IS',loss)
			print('LOSS IS',loss.shape)
			return loss.detach().numpy(), z.detach().numpy()[0]
	else:
		print('DATA IN BACK PROP IS:',data)
		y_pred = model(data)
		print('Y PRED IN BACK PROP IS:',y_pred)
		print('Y PRED SHAPE IS:',y_pred.shape)
		# Pause the execution and wait for user input
		input("Press Enter to continue...")

		loss = l(y_pred, data)
		if training:
			tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			return loss.detach().numpy(), y_pred.detach().numpy()


if __name__ == '__main__':
	train_loader, test_loader, labels = load_dataset(args.dataset)
	if args.model in ['MERLIN']:
		eval(f'run_{args.model.lower()}(test_loader, labels, args.dataset)')
	model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])

	## Prepare data
	trainD, testD = next(iter(train_loader)), next(iter(test_loader))
	trainO, testO = trainD, testD
	if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN'] or 'TranAD' in model.name: 
		trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)

	### Training phase
	if not args.test:
		print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
		num_epochs = 3; e = epoch + 1; start = time()
		for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
			lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
			accuracy_list.append((lossT, lr))
		print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
		save_model(model, optimizer, scheduler, e, accuracy_list)
		plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

	### Testing phase
	labels=testlabels.T
	#print('labels on 344 shape is',labels.shape)
	torch.zero_grad = True
	model.eval()
	print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
	loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
	
	###Plot curves
	if not args.test:
		if 'TranAD' in model.name: testO = torch.roll(testO, 1, 0) 
		print('TEST O IS:',testO)
		print('Y PRED IS:',y_pred)
		plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)
	
	### Scores
	df = pd.DataFrame()
	lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
	accumulated_scores = np.array([])
	accumulated_noise_scores=np.array([])
	noise_scores=np.array([])
	min_top_score=np.array([])
	for i in range(loss.shape[1]):
	    lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
	    #print("Loss T IS", lt.shape)
	    #print(lt)
	    #print("Loss IS", l.shape)
	    #print(l)
	    #print("Labels IS", ls.shape)
	   # print(ls)
	    #write_anomaly_to_file(score,label,file_path)
	    updated_scores, noise_scores,min_top_score = pot_scores(lt, l, ls)
        
        # Flatten updated_scores if it's multidimensional
	    #updated_scores = np.ravel(updated_scores)
	    #noise_scores=np.ravel(noise_scores)
        # Append the updated scores to the accumulated_scores array
	    accumulated_scores = np.concatenate((accumulated_scores, updated_scores))
	    #accumulated_noise_scores= np.concatenate((accumulated_noise_scores,test))
	#    print('THE MIN TOP SCORE IS',min_top_score)
	
	signal_prediction=[]
	classification=[]
	correct_pred_count=[]
	FAR_count=[]
	False_alarms=[]
	for i in range(loss.shape[1]):
	    result, pred , classification,correct_count,FAR_count= pot_eval(min_top_score,lt, l, ls)

	    if isinstance(result, dict):
	        # Handle result if it's a dictionary
	        # Example: Convert dictionary to DataFrame with proper columns
	        result_df = pd.DataFrame.from_dict(result, orient='index', columns=['Column_Name'])
	    elif isinstance(result, (list, np.ndarray)):
	        # Handle result if it's a list or numpy array
	        # Example: Convert list or numpy array to DataFrame with proper columns
	        result_df = pd.DataFrame(result, columns=['Column_Name'])
	    elif isinstance(result, pd.DataFrame):
	        # If result is already a DataFrame, use it directly
	        result_df = result
	    else:
	        # Handle other cases as needed
	        raise ValueError("Unsupported type for result")
	
	    df = pd.concat([df, result_df], ignore_index=True)
	for i in range(loss.shape[1]):
	    lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
	    print(i)
	   # print('loss is',loss.shape)
	 #   print('labels is',len(labels[:, i]))
	    result, pred,classification,correct_count,False_alarm = pot_eval(min_top_score,lt, l, ls)
	    FAR_count=FAR_count+False_alarm
	    correct_pred_count.append(correct_count)
	    if isinstance(result, dict):
	       # print('its dict')
	        # Handle result if it's a dictionary
	        # Example: Convert dictionary to DataFrame with proper columns
	        result_df = pd.DataFrame.from_dict(result, orient='index', columns=['Column_Name'])
	    elif isinstance(result, (list, np.ndarray)):
	        # Handle result if it's a list or numpy array
	        # Example: Convert list or numpy array to DataFrame with proper columns
	        print('NUMPY')
	        result_df = pd.DataFrame(result, columns=['Column_Name'])
	    elif isinstance(result, pd.DataFrame):
	        # If result is already a DataFrame, use it directly
	        print('already df')
	        result_df = result
	    else:
	        # Handle other cases as needed
	        raise ValueError("Unsupported type for result")
	
	    df = pd.concat([df, result_df], ignore_index=True)
	
		# preds = np.concatenate([i.reshape(-1, 1) + 0 for i in preds], axis=1)
		# pd.DataFrame(preds, columns=[str(i) for i in range(10)]).to_csv('labels.csv')
	    lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
	    labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
       # print('lossTfinal is : ',lossTfinal.shape)
     #   print('lossfina; is : ',lossFinal.shape)
      #  print('labels final is : ',labelsFinal.shape)
	  #  result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
	#    result.update(hit_att(loss, labels))
	   # result.update(ndcg(loss, labels))
		# print(df)
	    print(result)
	    print("\n")
		# pprint(getresults2(df, result))
		# beep(4)
	# Count the number of correct classifications
	correct_pred_count_sum=np.sum(correct_pred_count)
	classification_rate = (correct_pred_count_sum / len(correct_pred_count)) * 100

	# Print the classification rate

	print(f'Correct classification rate: {classification_rate:.2f}%')
	print('false alarm rate',FAR_count)