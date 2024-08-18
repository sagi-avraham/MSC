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
	sorted_indices = np.argsort(fpr)
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

def load_dataset(dataset, index):
	global testlabels
	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')

	# Define base file names
	file_bases = ['train', 'test', 'labels', 'testlabels', 'coindata', 'coinlabels']
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
	test_loader = DataLoader(loaders[1], batch_size=loaders[1].shape[0])
	coin_loader = DataLoader(loaders[4], batch_size=loaders[2].shape[0])
	
	labels = loaders[2]
	testlabels = loaders[3].T
	coinlabels = loaders[5].T

	#print(f"Train Loader Sample: {next(iter(train_loader)).shape}")
	#print(f"Test Loader Sample: {next(iter(test_loader)).shape}")
	#print(f"Coin Loader Sample: {next(iter(coin_loader)).shape}")

	return train_loader, test_loader, labels, coin_loader, coinlabels

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
	accuracy_list = []  
	# Initialize list to store average loss and lr per true_epoch
	train_loader, test_loader, labels, coin_loader, coinlabels = load_dataset(args.dataset, 1)
	if args.batchtrain:
		start = time()
		model, optimizer, scheduler, epoch, accuracy_list, last_loss = load_model(args.model, labels.shape[1])
		for true_epoch in range(1, 20):
			lossT_accumulated = np.array([])  # Reset accumulation for each true_epoch
			lr_accumulated = np.array([])  # Reset learning rate accumulation for each true_epoch
	
			
	
			
			for i in range(1,5):
				#print(f"Loading dataset iteration {i}...")
				train_loader, test_loader, labels, coin_loader, coinlabels = load_dataset(args.dataset, i)
	
				trainD, testD, coinD = next(iter(train_loader)), next(iter(test_loader)), next(iter(coin_loader))
				trainO, testO, coinO = trainD, testD, coinD
				
				if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN'] or 'TranAD' in model.name:
					trainD, testD, coinD = convert_to_windows(trainD, model), convert_to_windows(testD, model), convert_to_windows(coinD, model)
				
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
		print('ELSE STATEMENT')
		model, optimizer, scheduler, epoch, accuracy_list, last_loss = load_model(args.model, labels.shape[1])
		testD, coinD =  next(iter(test_loader)), next(iter(coin_loader))
		testO, coinO = testD, coinD
		testD, coinD =convert_to_windows(testD, model), convert_to_windows(coinD, model)
	### Testing phase
	labels=testlabels.T
	coinlabels=coinlabels.T
	#print('labels on 344 shape is',labels.shape)
	torch.zero_grad = True
	model.eval()
	print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
	loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
	loss_coin, y_pred_coin = backprop(0, model, coinD, coinO, optimizer, scheduler, training=False)
	
	###Plot curves
	if args.test:
		if 'TranAD' in model.name: testO = torch.roll(testO, 1, 0) 
		#print('TEST O IS:',testO)
		#print('Y PRED IS:',y_pred)
		plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)
		
	if args.test:
		if 'TranAD' in model.name: coinO = torch.roll(coinO, 1, 0) 
		#print('TEST O IS:',testO)
		#print('Y PRED IS:',y_pred)

		plotter(f'{args.model}_{args.dataset} COIN', coinO, y_pred_coin, loss_coin, coinlabels)
	
	### Scores
	df = pd.DataFrame()
	df_coin = pd.DataFrame()
	lossT, _ = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
	lossT_coin, _ = backprop(0, model, coinD, coinO, optimizer, scheduler, training=False)
	accumulated_scores = np.array([])
	accumulated_noise_scores=[]
	noise_scores=np.array([])
	min_top_score=np.array([])

	accumulated_scores_coin = np.array([])
	accumulated_noise_scores_coin=np.array([])
	noise_scores_coin=np.array([])
	min_top_score_coin=np.array([])
	fraction=1
	for i in range(loss.shape[1]):
		lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
		lt_coin, l_coin, ls_coin = lossT_coin[:, i], loss_coin[:, i], coinlabels[:, i]
		#print("Loss T IS", lt.shape)
		#print(lt)
		#print("Loss IS", l.shape)
		#print(l)
		#print("Labels IS", ls.shape)
	   # print(ls)
		#write_anomaly_to_file(score,label,file_path)
		updated_scores, noise_scores,min_top_score = pot_scores(lt, l, ls)
		print('len noise_score',len(noise_scores))
		accumulated_noise_scores.extend(noise_scores)
		#input('21312')
	 #   accumulated_noise_scores.append(noise_scores)
		updated_scores_coin, noise_scores_coin,min_top_score_coin = pot_scores(lt_coin, l_coin, ls_coin)
		#min_top_score_coin=min_top_score
		# Flatten updated_scores if it's multidimensional
		#updated_scores = np.ravel(updated_scores)
		#noise_scores=np.ravel(noise_scores)
		# Append the updated scores to the accumulated_scores array
		accumulated_scores = np.concatenate((accumulated_scores, updated_scores))
		accumulated_scores_coin = np.concatenate((accumulated_scores_coin, updated_scores_coin))
		#accumulated_noise_scores= np.concatenate((accumulated_noise_scores,test))
	#	print('THE MIN TOP SCORE IS',min_top_score)
	#accumulated_noise_scores=np.ravel(accumulated_noise_scores)
	#print('accumulatee noise scores is',len(accumulated_noise_scores))
	# Sort the scores array in descending order
	
	
	sorted_scores = np.sort(accumulated_noise_scores)[::-1]

	for i in range(loss.shape[1]):
			result, pred , classification,correct_count,FAR_count,signal_prediction,TP,FP,TN,FN,true_positive_segment,false_positive_segment,true_negative_segment,false_negative_segment= pot_eval(min_top_score,lt, l, ls)
			result_coin, pred , classification_coin,correct_count_coin,FAR_count_coin,signal_prediction_coin,TP,FP,TN,FN,true_positive_segment,false_positive_segment,true_negative_segment,false_negative_segment= pot_eval(min_top_score,lt_coin, l_coin, ls_coin)
	
			if isinstance(result, dict):
				# Handle result if it's a dictionary
				# Example: Convert dictionary to DataFrame with proper columns
				result_df = pd.DataFrame.from_dict(result, orient='index', columns=['Column_Name'])
				result_df_coin = pd.DataFrame.from_dict(result_coin, orient='index', columns=['Column_Name'])
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
			df_coin = pd.concat([df_coin, result_df_coin], ignore_index=True)
		# Create and open the file in write mode, this will overwrite the file if it already exists
file_path = 'run_statistics.txt'

# Open the file and write the header information
with open(file_path, 'w') as file:
	file.write("Run Statistics Log\n")
	file.write("===================\n\n")

# Now, append the logs during each iteration
with open(file_path, 'a') as file:
	for j in range(40):
		signal_prediction = []
		classification = []
		correct_pred_count = []
		FAR_count = 0  # Initialize as an integer
		False_alarms = []

		signal_prediction_coin = []
		classification_coin = []
		correct_pred_count_coin = []
		FAR_count_coin = 0  # Initialize as an integer
		False_alarms_coin = []
		combined_correct_pred_count = 0
		combined_far = 0
		true_positive_rate = np.zeros((40,))
		false_positive_rate = np.zeros((40,))
		true_positive_count = 0
		false_positive_count = 0
		true_negative_count = 0
		false_negative_count = 0

		true_positive_segment = 0
		false_positive_segment = 0
		true_negative_segment = 0
		false_negative_segment = 0
		TP = []
		FP = []
		TN = []
		FN = []

		top_count = max(1, int(len(sorted_scores) * fraction))
		top_scores = sorted_scores[:top_count]
		min_top_score = np.min(top_scores)

		file.write(f"Iteration {j+1} - Min Top Score: {min_top_score}\n")

		for i in range(loss.shape[1]):
			lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
			lt_coin, l_coin, ls_coin = lossT_coin[:, i], loss_coin[:, i], coinlabels[:, i]

			result, pred, classification, correct_count, False_alarm, signal_prediction, TP, FP, TN, FN, true_positive_segment, false_positive_segment, true_negative_segment, false_negative_segment = pot_eval(min_top_score, lt, l, ls)
			result_coin, pred_coin, classification_coin, correct_count_coin, False_alarm_coin, signal_prediction_coin, TP, FP, TN, FN, true_positive_segment, false_positive_segment, true_negative_segment, false_negative_segment = pot_eval(min_top_score, lt_coin, l_coin, ls_coin)

			FAR_count += False_alarm  # Increment by the integer value
			FAR_count_coin += False_alarm_coin  # Increment by the integer value

			correct_pred_count.append(correct_count)
			correct_pred_count_coin.append(correct_count_coin)

			true_positive_count += true_positive_segment
			false_positive_count += false_positive_segment
			true_negative_count += true_negative_segment
			false_negative_count += false_negative_segment

			is_label_one = np.any(labels[:, i] == 1)
			is_label_zero = np.all(labels[:, i] == 0)

			if is_label_one:
				if signal_prediction == 1 and signal_prediction_coin == 1:
					combined_correct_pred_count += 1
			elif is_label_zero:
				if signal_prediction == 0 and signal_prediction_coin == 0:
					combined_correct_pred_count += 1
				elif signal_prediction == 1 and signal_prediction_coin == 1:
					combined_far += 1

			correct_pred_count_sum = np.sum(correct_pred_count)
			classification_rate = (correct_pred_count_sum / 10) * 100
			combined_classification_rate = (combined_correct_pred_count / 10) * 100
			correct_pred_count_sum_coin = np.sum(correct_pred_count_coin)
			classification_rate_coin = (correct_pred_count_sum_coin / 10) * 100
			
			if isinstance(result, dict):
				result_df = pd.DataFrame.from_dict(result, orient='index', columns=['Column_Name'])
				result_df_coin = pd.DataFrame.from_dict(result_coin, orient='index', columns=['Column_Name'])

			df = pd.concat([df, result_df], ignore_index=True)
			df_coin = pd.concat([df_coin, result_df_coin], ignore_index=True)

		fraction /= 1.3
		true_positive_rate[j] = true_positive_count / (true_positive_count + false_negative_count)
		false_positive_rate[j] = false_positive_count / (false_positive_count + true_negative_count)

		file.write(f'TP: {true_positive_count}, TN: {true_negative_count}, FP: {false_positive_count}, FN: {false_negative_count}\n')
		file.write(f'TRUE POSITIVE RATE: {true_positive_rate[j]:.4f}\n')
		file.write(f'FALSE POSITIVE RATE: {false_positive_rate[j]:.4f}\n')
		file.write(f'Correct classification rate: {classification_rate:.2f}%\n')
		file.write(f'False alarm rate: {FAR_count}\n')
		file.write(f'Correct classification rate for coincidence data: {classification_rate_coin:.2f}%\n')
		file.write(f'False alarm rate for coincidence data: {FAR_count_coin}\n')
		file.write(f'Combined correct classification rate: {combined_classification_rate:.2f}%\n')
		file.write(f'Combined false alarms: {combined_far}\n')
		file.write('\n')

			
			#input('enter to continue')
	compute_and_plot_roc(true_positive_rate, false_positive_rate, ROCFilename)
	print(f'Correct classification rate: {classification_rate:.2f}%')
	print('false alarm rate',FAR_count)
	
	print(f'Correct classification rate for coincidence data: {classification_rate_coin:.2f}%')
	print('false alarm rate for coincidence data',FAR_count)
	
	print(f'combined correct classification rate: {combined_classification_rate:.2f}%')
	print('combined false alarms is',combined_far)
		# Count the number of correct classifications
	
	print(f'Correct classification rate: {classification_rate:.2f}%')
	print('false alarm rate',FAR_count)
	
	print(f'Correct classification rate for coincidence data: {classification_rate_coin:.2f}%')
	print('false alarm rate for coincidence data',FAR_count)
	
	print(f'combined correct classification rate: {combined_classification_rate:.2f}%')
	print('combined false alarms is',combined_far)
		