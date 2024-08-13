import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import torch
import numpy as np

plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2

os.makedirs('plots', exist_ok=True)

def smooth(y, box_pts=1):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plotter(name, y_true, y_pred, ascore, labels):
    if 'TranAD' in name: 
        y_true = torch.roll(y_true, 1, 0)
    os.makedirs(os.path.join('plots', name), exist_ok=True)
    pdf = PdfPages(f'plots/{name}/output.pdf')
    
    for dim in range(10):
        if y_true.shape[1] <= dim or y_pred.shape[1] <= dim or labels.shape[1] <= dim or ascore.shape[1] <= dim:
            print(f"Skipping dimension {dim} due to insufficient size.")
            continue
        
        if np.any(labels[:, dim] == 1):
            signal = "label : TRUE SIGNAL"
        else:
            signal = "label : NO SIGNAL"
        
        y_t, y_p, l, a_s = y_true[:, dim], y_pred[:, dim], labels[dim, :], ascore[:, dim]
        
        # Scale the values by 0.5e-22
        scale_factor = 0.5e-22
        y_t_scaled = y_t * scale_factor
        y_p_scaled = y_p * scale_factor
        
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.set_ylabel('Strain')
        ax1.set_title(f'{signal}')
        
        ax1.plot(smooth(y_t_scaled), linewidth=0.2, label='True')
        ax1.plot(smooth(y_p_scaled), '-', alpha=0.6, linewidth=0.3, label='Predicted')
        
        ax3 = ax1.twinx()
        ax3.plot(l, '--', linewidth=0.3, alpha=0.5)
        ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3)
        
        # Highlight the region where the signal is present
        if signal == "label : TRUE SIGNAL":
            signal_region = np.where(labels[:, dim] == 1)[0]
            if len(signal_region) > 0:
                start_idx = signal_region[0]
                end_idx = signal_region[-1]
                ax1.axvspan(start_idx, end_idx, color='red', alpha=0.3)
        
        if dim == 0: 
            ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
        
        ax2.plot(smooth(a_s), linewidth=0.2, color='g')
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Anomaly Score')
        pdf.savefig(fig)
        plt.close()
    
    pdf.close()

# Example call to the plotter function
# Note: Make sure y_true, y_pred, ascore, and labels are properly defined tensors or numpy arrays before calling plotter
# plotter('example', y_true, y_pred, ascore, labels)