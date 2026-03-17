import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data from README
gcn_data = {
    'Algorithm': ['None', 'GA', 'PSO', 'ABC', 'SA', 'HC', 'RS', 'ACO', 'BO', 'Grid Search', 'Optuna'],
    'F1': [0.8378, 0.8506, 0.8506, 0.8496, 0.8435, 0.8430, 0.8493, 0.8419, 0.8504, 0.8497, 0.8512],
    'AUC': [0.9097, 0.9125, 0.7790, 0.8831, 0.7841, 0.9109, 0.9147, 0.9145, 0.9154, 0.9151, 0.9166],
    'Loss': [1.3414, 1.2876, 1.3920, 1.4161, 1.3751, 1.3705, 1.2535, 1.2698, 1.2674, 1.2031, 1.2522],
    'NDCG': [0.9910, 0.9913, 0.9680, 0.9885, 0.9685, 0.9914, 0.9920, 0.9918, 0.9911, 0.9920, 0.9920]
}

gan_data = {
    'Algorithm': ['None', 'GA', 'PSO', 'ABC', 'SA', 'HC', 'RS', 'ACO', 'BO', 'Grid Search', 'Optuna'],
    'F1': [0.7337, 0.7538, 0.7571, 0.7545, 0.7584, 0.7559, 0.7541, 0.7167, 0.7505, 0.7428, 0.7560],
    'AUC': [0.7528, 0.7772, 0.7781, 0.7773, 0.7583, 0.7773, 0.7752, 0.7353, 0.7743, 0.7630, 0.7817],
    'Avg Loss': [0.1522, 0.0088, 0.0147, 0.0421, -0.0260, 0.0015, 0.0432, 0.0170, 0.2257, 0.0554, 0.2407],
    'NDCG': [0.9699, 0.9723, 0.9706, 0.9724, 0.9699, 0.9726, 0.9725, 0.9649, 0.9725, 0.9712, 0.9738]
}

def plot_metrics(data_dict, title, loss_key, filename, y_min=None):
    df = pd.DataFrame(data_dict)
    
    # Set 'Algorithm' as index
    df.set_index('Algorithm', inplace=True)
    
    # Invert to match requested bottom-up display order, if desired, but typical
    # vertical bars we just read left to right.
    
    # Plotting
    # We want vertical bars now per the reviewer's comment: "Given that the horizontal space is much larger than the vertical one, wouldn't it be better to show vertical bars..."
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    metrics = ['F1', 'AUC', loss_key, 'NDCG']
    labels = df.index
    x = np.arange(len(labels))
    width = 0.2
    
    bars_f1 = ax.bar(x - 1.5*width, df['F1'], width, label='F1', color='purple')
    bars_auc = ax.bar(x - 0.5*width, df['AUC'], width, label='AUC', color='orange')
    bars_loss = ax.bar(x + 0.5*width, df[loss_key], width, label='Loss' if loss_key == 'Loss' else 'Avg Loss', color='green')
    bars_ndcg = ax.bar(x + 1.5*width, df['NDCG'], width, label='NDCG', color='dodgerblue')
    
    ax.set_ylabel('Scores / Loss')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1))

    # Add a horizontal line at y=0 instead of x=-0.2 (as requested "delete the vertical line for x=-0.2 in Fig. 3")
    # Actually, the reviewer asked to remove "delete the vertical line for x=-0.2" from Fig 3 (GAN),
    # which in a vertical bar chart would be a horizontal line at y=-0.2.
    # We just don't add any line at -0.2. We can add one at 0 to clarify positive vs negative limits.
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    
    if y_min is not None:
        ax.set_ylim(bottom=y_min)
        
    # Set grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# Generate GCN Plot (Figure 2)
plot_metrics(gcn_data, 'Comparing metrics for GCN', 'Loss', 'gcn_comparison_vertical.png')

# Generate GAN Plot (Figure 3)
# Make sure we don't have the -0.2 line. We'll let Matplotlib handle the bottom bound automagically (which will encompass negative Avg Loss like -0.3044).
plot_metrics(gan_data, 'Comparing metrics for GAN', 'Avg Loss', 'gan_comparison_vertical.png')

print("Figures 'gcn_comparison_vertical.png' and 'gan_comparison_vertical.png' generated successfully!")
