import numpy as np
from src.spot import SPOT
from src.constants import lm
from sklearn.metrics import roc_auc_score
import os
import time  # Import time for time-related functions

scores_array = []
file_path = 'anomaly/anomalyscores.txt'

def pot_scores(init_score, score, label, file_path='anomaly/anomalyscores.txt', q=1e-5, level=0.02):
    """
    Run POT method on given score.
    Args:
        init_score (np.ndarray): The data to get init threshold.
            It should be the anomaly score of the train set.
        score (np.ndarray): The data to run POT method.
            It should be the anomaly score of the test set.
        label (np.ndarray): The ground-truth labels.
        file_path (str): Path to the file where scores will be written.
        q (float): Detection level (risk).
        level (float): Probability associated with the initial threshold t.
    Returns:
        tuple: A tuple containing:
            - score (np.ndarray): The updated score array.
            - scores_array (list): A list of the scores that were written to the file.
            - min_top_score (float): The lowest value in the top percentile scores.
    """
    # print('Testing POT method...')

    fraction = 0.0000005
    print('fraction is', fraction)  # Define the fraction of top scores to consider

    lms = lm[0]  # Assuming lm is defined in src.constants
    scores_array = []  # Initialize the list to store scores

    while True:
        try:
            s = SPOT(q)  # SPOT object
            s.fit(init_score, score)  # Fit the SPOT model
            s.initialize(level=lms, min_extrema=False, verbose=False)  # Initialization step
        except Exception as e:
            print(f"Exception during SPOT initialization: {e}")
            lms = lms * 0.999  # Adjust the level if an error occurs
        else:
            break

    # Ensure `score` is a NumPy array
    score = np.asarray(score)
    
    # Sort the scores array in descending order
    sorted_scores = np.sort(score)[::-1]
    
    # Calculate the number of scores to consider for the top fraction
    top_count = max(1, int(len(sorted_scores) * fraction))
    
    # Get the top scores
    top_scores = sorted_scores[:top_count]
    
    # Determine the lowest value in the top scores
    min_top_score = np.min(top_scores)

    # Check if file exists and delete it if it's older than 30 seconds
    if os.path.exists(file_path):
        file_mod_time = os.path.getmtime(file_path)
        current_time = time.time()
        
        if (current_time - file_mod_time) > 30:
            os.remove(file_path)
    
    # Create directory if it does not exist
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    # Write scores to file only if all labels are 0
    if np.all(label == 0):
        with open(file_path, 'a') as file:
            for s in score:
                file.write(f"{s}\n")
                scores_array.append(s)  # Append score to the list

    print('min top score is', min_top_score)
    
    return score, scores_array, min_top_score
