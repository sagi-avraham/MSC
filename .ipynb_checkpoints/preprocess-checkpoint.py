import os
import pandas as pd
import numpy as np

output_folder = 'processed'
data_folder = 'data/STRAIN'

def normalize3(a, min_a=None, max_a=None):
    print("norm3")
    return a, min_a, max_a

def load_data(dataset):
    print("load data")
    if dataset != 'STRAIN':
        raise ValueError("This function is specifically for the STRAIN dataset.")

    folder = os.path.join(output_folder, dataset)
    os.makedirs(folder, exist_ok=True)

    # Define the number of files you have
    num_files = 10  # Adjust this as needed based on your dataset

    for i in range(1, num_files + 1):
        # Load and process train, test, labels, testlabels, coinlabels, coindata files
        train_file = os.path.join(data_folder, f'train{i}.csv')
        test_file = os.path.join(data_folder, f'test{i}.csv')
        labels_file = os.path.join(data_folder, f'train_labels{i}.csv')
        testlabels_file = os.path.join(data_folder, f'test_labels{i}.csv')
        coinlabels_file = os.path.join(data_folder, f'coincidence_test_labels{i}.csv')
        coindata_file = os.path.join(data_folder, f'coincidence_test{i}.csv')

        # Read CSV files
        train = pd.read_csv(train_file).values.astype(float)
        test = pd.read_csv(test_file).values.astype(float)
        labels = pd.read_csv(labels_file).values.astype(float)
        testlabels = pd.read_csv(testlabels_file).values.astype(float)
        coinlabels = pd.read_csv(coinlabels_file).values.astype(float)
        coindata = pd.read_csv(coindata_file).values.astype(float)

        # Normalize datasets
        train, min_a, max_a = normalize3(train)
        test, _, _ = normalize3(test, min_a, max_a)
        # Note: You might need to adjust normalization for labels, testlabels, coinlabels, coindata if necessary

        # Save the processed files
        for file_name, data in zip(['train', 'test', 'labels', 'testlabels', 'coinlabels', 'coindata'],
                                    [train, test, labels, testlabels, coinlabels, coindata]):
            np.save(os.path.join(folder, f'{file_name}{i}.npy'), data)
        
        print(f"Processed and saved files for index {i}")

if __name__ == '__main__':
    load_data('STRAIN')
