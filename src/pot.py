import numpy as np
from src.spot import SPOT
from src.constants import *
from sklearn.metrics import *
import os




global actual_label


def adjust_predicts(min_top_score,score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        print('len score',len(score))
        print('len label',len(label))
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
   # print('LABEL IS:',label)
    latency = 0
    if pred is None:
        predict = score > min_top_score
    #    print('min top score is',min_top_score)
    else:
        predict = pred

    actual = label
   # print('actual is', actual)
    #print('predict shape is', predict)
    #if np.any(actual == 1):
     #   print('signal')
    #else:
      #  print('no signal')
    anomaly_state = False
    anomaly_count = 0
    count=0
    for i in range(len(score)):
        if predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, i-50, -1):
                if score[i] < min_top_score:
                    count=+1
                    if count>25:
	                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict




def calc_point2point(predict, actual):
    """
    Calculate evaluation metrics by comparing predict and actual labels.
    Args:
        predict (np.ndarray): The predicted labels
        actual (np.ndarray): The actual labels
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    anomalies=np.sum(predict)
    try:
        roc_auc = roc_auc_score(actual, predict)
    except:
        roc_auc = 0

    return f1, precision, recall, TP, TN, FP, FN, roc_auc,anomalies





def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target = calc_seq(score, label, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    print(m, m_t)
    return m, m_t

def save_results(file_path, results, append=True):
    """
    Save or append evaluation results to a text file.

    Args:
        file_path (str): Path to the file where results will be saved.
        results (dict): Dictionary containing evaluation results.
        append (bool): Whether to append to the file if it exists. Defaults to True.
    """
    mode = 'a' if append else 'w'
    with open(file_path, mode) as f:
        if append and f.tell() == 0:
            # Write header if appending and file is empty
            header = "f1,precision,recall,TP,TN,FP,FN,ROC/AUC,threshold\n"
            f.write(header)
        # Write results
        line = (
            f"{results['f1']},{results['precision']},{results['recall']},"
            f"{results['TP']},{results['TN']},{results['FP']},{results['FN']},"
            f"{results['ROC/AUC']},{results['threshold']}\n"
        )
        f.write(line)

def pot_eval(min_top_score, init_score, score, label, q=1e-5, level=0.02):
    """
    Run POT method on given score.
    Args:
        init_score (np.ndarray): The data to get init threshold.
            It should be the anomaly score of the training set.
        score (np.ndarray): The data to run the POT method.
            It should be the anomaly score of the test set.
        label (np.ndarray): The actual labels.
        q (float): Detection level (risk).
        level (float): Probability associated with the initial threshold t.
    Returns:
        dict: POT result dictionary.
    """
    lms = lm[0]
    predict_label=[]
    False_alarm=0
    while True:
        try:
            s = SPOT(q)  # SPOT object
            s.fit(init_score, score)  # Data import
            
            s.initialize(level=lms, min_extrema=False, verbose=False)  # Initialization step
        except:
            lms = lms * 0.999
        else:
            break
    ret = s.run(dynamic=False)  # Run
    
    pot_th = np.mean(ret['thresholds']) * lm[1]
   # print('label in pot result is:', len(label))
    pred, p_latency = adjust_predicts(min_top_score, score, label, pot_th, calc_latency=True)
    p_t = calc_point2point(pred, label)
    
    # Define the criteria for signal status
    TP = p_t[3]
    FN = p_t[6]
    true_positive=0
    false_positive=0
    true_negative=0
    false_negative=0
    anomalies=p_t[8]
    # Determine signal status based on TP and FN
    if anomalies > 200:
        signal_prediction = 1
    else:
       
        signal_prediction = 0 

 # Determine actual_label based on the labels
    if np.any(label == 1):
        actual_label = 'Signal'
    else:
        actual_label = 'Noise'
		
    if signal_prediction == 1:
		
        if actual_label == 'Signal':
	        correct_pred_count = 1
	        predict_label = 'Signal'
	        true_positive=1
			
        elif actual_label == 'Noise':
	        correct_pred_count = 0
	        predict_label = 'Signal'
	        False_alarm=1
	        false_positive=1
    else:
		
        if actual_label == 'Noise':
	        correct_pred_count = 1
	        predict_label = 'Noise'
	        true_negative=1
			
        elif actual_label == 'Signal':
	        false_negative=1
	        correct_pred_count = 0
	        predict_label = 'Noise'

    results = {
        'f1': p_t[0],
        'precision': p_t[1],
        'recall': p_t[2],
        'TP': TP,
        'TN': p_t[4],
        'FP': p_t[5],
        'FN': FN,
        'ROC/AUC': p_t[7],
        'threshold': pot_th,
        'Signal prediction': predict_label,
        'True signal label': actual_label,
		'anomalies': p_t[8] #Add the signal status to the results
    }
    
    # Save results to a file
    # save_results('evaluation_results.txt', results)
    
    # Optionally save predictions to a text file
    # np.savetxt('predictions.txt', pred, fmt='%f', delimiter=',')
    
    # Print the signal status
   # print(f"Signal prediction: {signal_prediction}")
    print(min_top_score)
    return results, np.array(pred), actual_label,correct_pred_count,False_alarm,signal_prediction,TP,p_t[5],p_t[4],FN,true_positive,false_positive,true_negative,false_negative