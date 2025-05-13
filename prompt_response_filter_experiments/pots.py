import torch
import umap
import matplotlib.pyplot as plt
import matplotlib as mpl

# Load activations with shape (n_samples, layers, hidden_size)
activations = torch.load("activations/final_activations_prompts.pt")
n_samples, layers, hidden_size = activations.shape

# Create category: honest (0) for even-indexed samples, deceptive (1) for odd-indexed samples.
category = [0 if i % 2 == 0 else 1 for i in range(n_samples)]
category = torch.tensor(category)

# Define the grid layout: 6 rows x 8 columns
n_rows, n_cols = 6, 8

# Create a figure and axes grid
fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 18))

# Loop over each layer and plot its UMAP embedding in its corresponding subplot.
for layer_idx in range(layers):
    # Determine the subplot location in the grid.
    row = layer_idx // n_cols
    col = layer_idx % n_cols
    ax = axes[row, col]
    
    # Extract the activations for the current layer. Shape: [n_samples, hidden_size]
    X = activations[:, layer_idx, :].numpy()
    
    # Initialize the UMAP reducer (for a 2D embedding)
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(X)
    
    # Scatter the points in the subplot; color each point by its category.
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=category.numpy(),
        cmap='coolwarm',
        alpha=0.7
    )
    
    # Annotate the subplot.
    ax.set_title(f"Layer {layer_idx}")
    ax.set_xticks([])
    ax.set_yticks([])

# If the total number of subplots in the grid exceeds the number of layers, remove extra axes.
total_plots = n_rows * n_cols
if layers < total_plots:
    for idx in range(layers, total_plots):
        fig.delaxes(axes.flatten()[idx])

# Create a single global colorbar.
norm = mpl.colors.Normalize(vmin=0, vmax=1)
sm = mpl.cm.ScalarMappable(norm=norm, cmap="coolwarm")
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), ticks=[0, 1])
cbar.ax.set_yticklabels(['honest', 'deceptive'])
cbar.set_label("Category")

plt.tight_layout()
plt.savefig("umap_grid.png")
plt.close(fig)




