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
import asyncio
from pprint import pprint
# from beepy import beep
import numpy as np
from src.spot import SPOT
from src.constants import *
from sklearn.metrics import *
import os
ROCFilename='ROC PLOT'
output_filename='ROC PLOT'
import asyncio

async def main():
	for i in range(0,1):
		print(f"Iteration {i} - Waiting for 5 seconds...")
		await asyncio.sleep(3)  # Await the sleep to actually pause execution
		print(f"Iteration {i} - Done waiting!")
	
def train_batches(model, optimizer, scheduler, train_loader, num_batches):
	for batch_idx in range(num_batches):
		print(f"Training batch {batch_idx + 1}/{num_batches}")
		for data in train_loader:
			data = data.double()
			loss, lr = backprop(epoch=batch_idx, model=model, data=data, dataO=data, optimizer=optimizer, scheduler=scheduler, training=True)
		
		# Save model after each batch
		save_model(model, optimizer, scheduler, batch_idx, accuracy_list=[])
		print(f"Finished training batch {batch_idx + 1}/{num_batches}\nModel saved.\n")


def compute_and_plot_roc(true_positive_rate, false_positive_rate, ROCFilename):
	"""
	Computes and plots the ROC curve from true positive rate and false positive rate.
	
	Parameters:
	- true_positive_rate: List or array of True Positive Rate values.
	- false_positive_rate: List or array of False Positive Rate values.
	- ROCFilename: Filename to save the ROC curve plot.
	"""
	# Convert to numpy arrays for easier manipulation
	tpr = np.array(true_positive_rate)
	fpr = np.array(false_positive_rate)
	
	# Sort FPR and corresponding TPR values
	#sorted_indices = np.argsort(fpr)
	fpr_sorted = fpr
	tpr_sorted = tpr
	
	# Compute ROC curve (area under the curve)
	roc_auc = auc(fpr_sorted, tpr_sorted)
	
	# Plot ROC curve
	plt.figure()
	plt.plot(fpr_sorted, tpr_sorted, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic')
	plt.legend(loc='lower right')
	plt.grid(True)
	plt.savefig(ROCFilename, format='pdf')
	plt.close()
	
def convert_to_windows(data, model):
	windows = []; w_size = model.n_window
	
	for i, g in enumerate(data): 
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w if 'TranAD' in args.model or 'Attention' in args.model else w.view(-1))
	return torch.stack(windows)

def load_dataset_train(dataset, index):
	global testlabels
	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')

	# Define base file names
	file_bases = ['test', 'labels']
	loaders = []

	# Load each file iteratively based on index
	for file_base in file_bases:
		file_name = f"{file_base}{index}.npy"  # e.g., 'train1.npy', 'test1.npy'
		file_path = os.path.join(folder, file_name)
		
		if not os.path.exists(file_path):
			raise Exception(f"File {file_name} not found in {folder}.")
		
		data = np.load(file_path, allow_pickle=True, mmap_mode='r')
		loaders.append(data)
	
	# Apply less data option if needed
	if args.less:
		loaders[0] = cut_array(0.2, loaders[0])  # Assuming `loader[0]` is train data

	# Create DataLoader instances
	train_loader = DataLoader(loaders[0], batch_size=loaders[0].shape[0])	
	labels = loaders[1]


	#print(f"Train Loader Sample: {next(iter(train_loader)).shape}")
	#print(f"Test Loader Sample: {next(iter(test_loader)).shape}")
	#print(f"Coin Loader Sample: {next(iter(coin_loader)).shape}")

	return train_loader,labels

def load_dataset_test(dataset, index):
	global testlabels
	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')

	# Define base file names
	file_bases = ['test', 'testlabels', 'coindata', 'coinlabels']
	loaders = []

	# Load each file iteratively based on index
	for file_base in file_bases:
		file_name = f"{file_base}{index}.npy"  # e.g., 'train1.npy', 'test1.npy'
		file_path = os.path.join(folder, file_name)
		
		if not os.path.exists(file_path):
			raise Exception(f"File {file_name} not found in {folder}.")
		
		data = np.load(file_path, allow_pickle=True, mmap_mode='r')
		loaders.append(data)
	
	# Apply less data option if needed
	if args.less:
		loaders[0] = cut_array(0.2, loaders[0])  # Assuming `loader[0]` is train data

	# Create DataLoader instances
	test_loader = DataLoader(loaders[0], batch_size=loaders[0].shape[0])
	coin_loader = DataLoader(loaders[2], batch_size=loaders[2].shape[0])
	
	testlabels = loaders[1].T
	coinlabels = loaders[3].T

	#print(f"Train Loader Sample: {next(iter(train_loader)).shape}")
	#print(f"Test Loader Sample: {next(iter(test_loader)).shape}")
	#print(f"Coin Loader Sample: {next(iter(coin_loader)).shape}")

	return test_loader, testlabels, coin_loader, coinlabels

def save_model(model, optimizer, scheduler, epoch, accuracy_list, last_loss):
	folder = f'checkpoints/{args.model}_{args.dataset}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model.ckpt'
	torch.save({
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'scheduler_state_dict': scheduler.state_dict(),
		'accuracy_list': accuracy_list,
		'last_loss': last_loss  # Save the last loss
	}, file_path)


def load_model(modelname, dims):
	import src.models
	model_class = getattr(src.models, modelname)
	model = model_class(dims).double()

	optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1e-3, 0.9)
	fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
	
	if os.path.exists(fname) and (not args.retrain or args.test):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(fname)
		
		# Load the model, optimizer, scheduler states
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
		last_loss = checkpoint.get('last_loss', None)  # Load the last loss

		# Print the loaded states for debugging or inspection
		#print(f"Model State Dict: {model.state_dict()}")
		#print(f"Optimizer State Dict: {optimizer.state_dict()}")
		#print(f"Scheduler State Dict: {scheduler.state_dict()}")
		#print(f"Epoch: {epoch}")
		#print(f"Accuracy List: {accuracy_list}")
		#print(f"Last Loss: {last_loss}")
		
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1
		accuracy_list = []
		last_loss = None  # Initialize last loss
	
	return model, optimizer, scheduler, epoch, accuracy_list, last_loss

	
	return model, optimizer, scheduler, epoch, accuracy_list, last_loss
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
			#tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			for d, _ in dataloader:
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, bs, feats)
				z = model(window, elem)
				if isinstance(z, tuple): z = z[1]
			loss = l(z, elem)[0]
		##	print('LOSS IS',loss)
		#	print('LOSS IS',loss.shape)
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
	if args.batchtrain:
		accuracy_list = []  
		# Initialize list to store average loss and lr per true_epoch
		train_loader, labels = load_dataset_train(args.dataset, 1)
		start = time()
		model, optimizer, scheduler, epoch, accuracy_list, last_loss = load_model(args.model, labels.shape[1])
		for true_epoch in range(1, 100):
			lossT_accumulated = np.array([])  # Reset accumulation for each true_epoch
			lr_accumulated = np.array([])  # Reset learning rate accumulation for each true_epoch
	
			
	
			
			for i in range(1,100):
				#print(f"Loading dataset iteration {i}...")
				train_loader, labels= load_dataset_train(args.dataset, i)
	
				trainD= next(iter(train_loader))
				trainO = trainD
				
				if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN'] or 'TranAD' in model.name:
					trainD = convert_to_windows(trainD, model), 
				
				if not args.test:
					#print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
					num_epochs = 1
				
					for e in tqdm(range(epoch + 1, epoch + num_epochs + 1)):
						lossT, lr = backprop(true_epoch, model, trainD, trainO, optimizer, scheduler)
						
						# Accumulate lossT and lr
						lossT_accumulated = np.append(lossT_accumulated, lossT)
						lr_accumulated = np.append(lr_accumulated, lr)
						
						last_loss = lossT
					
			
			# After the loop, calculate the average loss and lr for this true_epoch
			avg_lossT = np.mean(lossT_accumulated)
			avg_lr = np.mean(lr_accumulated)
			accuracy_list.append((avg_lossT, avg_lr))  # Store average loss and learning rate for plotting
			print('Epoch :',true_epoch,'	Loss :',avg_lossT)
			print('\n')
			
		# Plot the accuracies (average losses)
		print(color.BOLD + 'Training time: ' + "{:10.4f}".format(time() - start) + ' s' + color.ENDC)
		plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')
		save_model(model, optimizer, scheduler, e, accuracy_list, last_loss)

	else:
		print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
		test_loader, testlabels = load_dataset_train(args.dataset, 1)
		model, optimizer, scheduler, epoch, accuracy_list, last_loss = load_model(args.model, testlabels.shape[1])
		import numpy as np

		# Initialize arrays for batch, coin, and combined counts with zeros
		false_positive_count_batch = np.zeros((100, 40))
		false_negative_count_batch = np.zeros((100, 40))
		true_positive_count_batch = np.zeros((100, 40))
		true_negative_count_batch = np.zeros((100, 40))
		
		false_positive_count_coin_batch = np.zeros((100, 40))
		false_negative_count_coin_batch = np.zeros((100, 40))
		true_positive_count_coin_batch = np.zeros((100, 40))
		true_negative_count_coin_batch = np.zeros((100, 40))
		
		false_positive_count_combined_batch = np.zeros((100, 40))
		false_negative_count_combined_batch = np.zeros((100, 40))
		true_positive_count_combined_batch = np.zeros((100, 40))
		true_negative_count_combined_batch = np.zeros((100, 40))

		threshold_score=np.zeros(40)
		for batch in range(1, 5):
		    test_loader, testlabels, coin_loader, coinlabels = load_dataset_test(args.dataset, batch)
		    testD, coinD = next(iter(test_loader)), next(iter(coin_loader))
		    testO, coinO = testD, coinD
		    testD, coinD = convert_to_windows(testD, model), convert_to_windows(coinD, model)
		    
		    ### Testing phase
		    labels = testlabels.T
		    coinlabels = coinlabels.T
		    torch.zero_grad = True
		    model.eval()
		    print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
		    
		    loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
		    loss_coin, y_pred_coin = backprop(0, model, coinD, coinO, optimizer, scheduler, training=False)
		    
		    ### Plot curves
		    if args.test:
		        if 'TranAD' in model.name:
		            testO = torch.roll(testO, 1, 0)
		        plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)
		        
		    if args.test:
		        if 'TranAD' in model.name:
		            coinO = torch.roll(coinO, 1, 0)
		        
		        df = pd.DataFrame()
		        df_coin = pd.DataFrame()
		        
		        lossT, _ = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
		        lossT_coin, _ = backprop(0, model, coinD, coinO, optimizer, scheduler, training=False)
		        
		        accumulated_scores = np.array([])
		        accumulated_noise_scores = []
		        noise_scores = np.array([])
		        min_top_score = np.array([])
		        
		        accumulated_scores_coin = np.array([])
		        accumulated_noise_scores_coin = np.array([])
		        noise_scores_coin = np.array([])
		        min_top_score_coin = np.array([])
		        fraction = 1
		        
		        for i in range(loss.shape[1]):
		            lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
		            lt_coin, l_coin, ls_coin = lossT_coin[:, i], loss_coin[:, i], coinlabels[:, i]
		            updated_scores, noise_scores, min_top_score = pot_scores(lt, l, ls)
		            accumulated_noise_scores.extend(noise_scores)
		            updated_scores_coin, noise_scores_coin, min_top_score_coin = pot_scores(lt_coin, l_coin, ls_coin)
		            accumulated_scores = np.concatenate((accumulated_scores, updated_scores))
		            accumulated_scores_coin = np.concatenate((accumulated_scores_coin, updated_scores_coin))
		        
		        sorted_scores = np.sort(accumulated_noise_scores)[::-1]
		        
		        for i in range(loss.shape[1]):
		            pred, signal_prediction, true_positive_segment, false_positive_segment, true_negative_segment, false_negative_segment = pot_eval(min_top_score, lt, l, ls)
		            pred, signal_prediction, true_positive_segment, false_positive_segment, true_negative_segment, false_negative_segment = pot_eval(min_top_score, lt_coin, l_coin, ls_coin)
		        
		        true_positive_rate = np.zeros((40,))
		        false_positive_rate = np.zeros((40,))
		        true_negative_rate = np.zeros((40,))
		        false_negative_rate = np.zeros((40,))
		        precision = np.zeros((40,))
		    
		        true_positive_rate_coin = np.zeros((40,))
		        false_positive_rate_coin = np.zeros((40,))
		        false_negative_rate_coin = np.zeros((40,))
		        true_negative_rate_coin = np.zeros((40,))
		        precision_coin = np.zeros((40,))
		    
		        true_positive_rate_combined = np.zeros((40,))
		        false_positive_rate_combined = np.zeros((40,))
		        true_negative_rate_combined = np.zeros((40,))
		        false_negative_rate_combined = np.zeros((40,))
		        precision_combined = np.zeros((40,))
		        
		        for j in range(20):
		            signal_prediction = []
		            correct_pred_count = []
		        
		            signal_prediction_coin = []
		            correct_pred_count_coin = []
		        
		            true_positive_count = 0
		            false_positive_count = 0
		            true_negative_count = 0
		            false_negative_count = 0
		        
		            true_positive_count_coin = 0
		            false_positive_count_coin = 0
		            true_negative_count_coin = 0
		            false_negative_count_coin = 0
		        
		            true_positive_count_combined = 0 
		            false_positive_count_combined = 0
		            true_negative_count_combined = 0
		            false_negative_count_combined = 0
		        
		            true_positive_combined = 0 
		            false_positive_combined = 0
		            true_negative_combined = 0
		            false_negative_combined = 0
		        
		            true_positive_segment = 0
		            false_positive_segment = 0
		            true_negative_segment = 0
		            false_negative_segment = 0
		        
		            true_positive_segment_coin = 0
		            false_positive_segment_coin = 0
		            true_negative_segment_coin = 0
		            false_negative_segment_coin = 0
		        
		            top_count = max(1, int(len(sorted_scores) * fraction))
		            top_scores = sorted_scores[:top_count]
		            min_top_score = np.min(top_scores)
		        
		            for i in range(loss.shape[1]):
		                lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
		                lt_coin, l_coin, ls_coin = lossT_coin[:, i], loss_coin[:, i], coinlabels[:, i]
		        
		                pred, signal_prediction, true_positive_segment, false_positive_segment, true_negative_segment, false_negative_segment = pot_eval(min_top_score, lt, l, ls)
		                pred_coin, signal_prediction_coin, true_positive_segment_coin, false_positive_segment_coin, true_negative_segment_coin, false_negative_segment_coin = pot_eval(min_top_score, lt_coin, l_coin, ls_coin)
		        
		                true_positive_count += true_positive_segment 
		                false_positive_count += false_positive_segment
		                true_negative_count += true_negative_segment
		                false_negative_count += false_negative_segment
		        
		                true_positive_count_coin += true_positive_segment_coin 
		                false_positive_count_coin += false_positive_segment_coin
		                true_negative_count_coin += true_negative_segment_coin
		                false_negative_count_coin += false_negative_segment_coin
		        
		                is_label_one = np.any(labels[:, i] == 1)
		                is_label_zero = np.all(labels[:, i] == 0)
		        
		                # (IF TEST HAS A TRUE SIGNAL THEN COINCIDENCE DATA WILL ALSO HAVE A TRUE SIGNAL)
		                # THIS WILL REDUCE THE RATES AT WHICH TRUE POSITIVE SIGNALS ARE DETECTED AS ONE MAY CORRECTLY IDENTIFY A SIGNAL BUT THE OTHER MAY NOT DETECT THE TRUE SIGNAL OVER ITS NOISE. HOWEVER THIS SHOULD REDUCE THE RATE OF FALSE POSITIVES.
		        
		                # If the true label is signal and both predict signal, this is a TP for combined data
		                # If the true label is signal and they aren't both predicted signals, then count this as a false negative for the combined data
		                # If true label is no signal and they both predict no signal, this is a TN for combined data
		                # If true label is no signal and either one predicts a signal, this is a false positive 
		                true_positive_combined = 0
		                false_negative_combined = 0
		                true_negative_combined = 0
		                false_positive_combined = 0
		                if is_label_one:
		                    print('SIGNAL')
		                    if signal_prediction == 1 and signal_prediction_coin == 1:
		                        true_positive_combined = 1
		                    else:
		                        false_negative_combined = 1
		                elif is_label_zero:
		                    print('NOISE')
		                    if signal_prediction == 0 and signal_prediction_coin == 0:
		                        true_negative_combined = 1
		                    if signal_prediction == 1 and signal_prediction_coin == 1: 
		                        false_positive_combined = 1
		        
		                true_positive_count_combined += true_positive_combined
		                false_positive_count_combined += false_positive_combined
		                false_negative_count_combined += false_negative_combined
		                true_negative_count_combined += true_negative_combined
		        
		            fraction /= 1.5  # Hang this to log scale 
		            print('TP COMBINED', true_positive_count_combined)
		            print('FP COMBINED', false_positive_count_combined)
		            print('TN COMBINED', true_negative_count_combined)
		            print('FN COMBINED', false_negative_count_combined)
		        
		            # Get TP, FP, TN, FN rates for test data
		            true_positive_rate[j] = true_positive_count / (true_positive_count + false_negative_count)
		            false_positive_rate[j] = false_positive_count / (false_positive_count + true_negative_count)
		            true_negative_rate[j] = true_negative_count / (true_negative_count + false_positive_count)
		            false_negative_rate[j] = false_negative_count / (false_negative_count + true_positive_count)
		        
		            if true_positive_count == 0:
		                precision[j] = 0
		            else:
		                precision[j] = true_positive_count / (true_positive_count + false_positive_count)
		        
		            # Get TP, FP, TN, FN rates for coin data
		            true_positive_rate_coin[j] = true_positive_count_coin / (true_positive_count_coin + false_negative_count_coin)
		            false_positive_rate_coin[j] = false_positive_count_coin / (false_positive_count_coin + true_negative_count_coin)
		            true_negative_rate_coin[j] = true_negative_count_coin / (true_negative_count_coin + false_positive_count_coin)
		            false_negative_rate_coin[j] = false_negative_count_coin / (false_negative_count_coin + true_positive_count_coin)
		            if true_positive_count_coin == 0:
		                precision[j] = 0
		            else:
		                precision_coin[j] = true_positive_count_coin / (true_positive_count_coin + false_positive_count_coin)
		        
		            # Get TP, FP, TN, FN rates for combined data
		            print('FP COUNT COMBINED, TN COUNT COMBINED', false_positive_count_combined, true_negative_count_combined)
		        
		            false_positive_count_batch[batch, j] = false_positive_count
		            false_negative_count_batch[batch, j] = false_negative_count
		            true_positive_count_batch[batch, j] = true_positive_count
		            true_negative_count_batch[batch, j] = true_negative_count
		        
		            false_positive_count_coin_batch[batch, j] = false_positive_count_coin
		            false_negative_count_coin_batch[batch, j] = false_negative_count_coin
		            true_positive_count_coin_batch[batch, j] = true_positive_count_coin
		            true_negative_count_coin_batch[batch, j] = true_negative_count_coin
		        
		            false_positive_count_combined_batch[batch, j] = false_positive_count_combined
		            false_negative_count_combined_batch[batch, j] = false_negative_count_combined
		            true_positive_count_combined_batch[batch, j] = true_positive_count_combined
		            true_negative_count_combined_batch[batch, j] = true_negative_count_combined
		            threshold_score[j] = min_top_score
		        
		            true_positive_rate_combined[j] = true_positive_count_combined / (true_positive_count_combined + false_negative_count_combined)
		            false_positive_rate_combined[j] = false_positive_count_combined / (false_positive_count_combined + true_negative_count_combined)
		            true_negative_rate_combined[j] = true_negative_count_combined / (true_negative_count_combined + false_positive_count_combined)
		            false_negative_rate_combined[j] = false_negative_count_combined / (false_negative_count_combined + true_positive_count_combined)
		        
		            if true_positive_count_combined == 0:
		                precision_combined[j] = 0
		            else:
		                precision_combined[j] = true_positive_count_combined / (true_positive_count_combined + false_positive_count_combined)
		
			
			# Sum all the arrays along axis 0 and overwrite the original variables
	false_positive_count_batch = np.sum(false_positive_count_batch, axis=0)
	false_negative_count_batch = np.sum(false_negative_count_batch, axis=0)
	true_positive_count_batch = np.sum(true_positive_count_batch, axis=0)
	true_negative_count_batch = np.sum(true_negative_count_batch, axis=0)
	
	false_positive_count_coin_batch = np.sum(false_positive_count_coin_batch, axis=0)
	false_negative_count_coin_batch = np.sum(false_negative_count_coin_batch, axis=0)
	true_positive_count_coin_batch = np.sum(true_positive_count_coin_batch, axis=0)
	true_negative_count_coin_batch = np.sum(true_negative_count_coin_batch, axis=0)
	
	false_positive_count_combined_batch = np.sum(false_positive_count_combined_batch, axis=0)
	false_negative_count_combined_batch = np.sum(false_negative_count_combined_batch, axis=0)
	true_positive_count_combined_batch = np.sum(true_positive_count_combined_batch, axis=0)
	true_negative_count_combined_batch = np.sum(true_negative_count_combined_batch, axis=0)

	tpr_batch=true_positive_count_batch/(true_positive_count_batch+false_negative_count_batch)
	fpr_batch=false_positive_count_batch/(false_positive_count_batch+true_negative_count_batch)
	print('TP,FN,FP,TN', true_positive_count_batch,false_negative_count_batch,false_positive_count_batch,true_negative_count_batch)
	file_path = 'run_statistics.txt'
	with open(file_path, 'w') as file:
		file.write("Run Statistics Log\n")
		file.write("===================\n\n")
		for j in range(1,40):
			file.write(f'Threshold Anomaly score: {threshold_score[j]}\n')
			file.write(f'TEST SET ----- TP: {true_positive_count_batch[j]}, TN: {true_negative_count_batch[j]}, FP: {false_positive_count_batch[j]}, FN: {false_negative_count_batch[j]}\n')
			file.write(f'COINCIDENCE SET ----- TP: {true_positive_count_coin_batch[j]}, TN: {true_negative_count_coin_batch[j]}, FP: {false_positive_count_coin_batch[j]}, FN: {false_negative_count_coin_batch[j]}\n')
			file.write(f'COMBINED SET ----- TP: {true_positive_count_combined_batch[j]}, TN: {true_negative_count_combined_batch[j]}, FP: {false_positive_count_combined_batch[j]}, FN: {false_negative_count_combined_batch[j]}\n')
			file.write(f'\n')
			
		compute_and_plot_roc(tpr_batch, fpr_batch, ROCFilename) #try and also plot the tpr rate for test and combined over each other so see how combined gives us better results, also try put this in log scale?
		
		
	