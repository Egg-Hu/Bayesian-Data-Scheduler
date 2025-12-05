


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
import torch  # For reading .pt files
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="")
parser.add_argument("--step", type=int, default=100)
parser.add_argument("--flag", type=str, default="all")
parser.add_argument("--transformation", type=str, default="softmax")
args = parser.parse_args()
if args.path=="":
    raise NotImplementedError
else:
    path=args.path
    step=args.step
    flag=args.flag
    transformation=args.transformation
with open(f"{path}/new_dataset.json", "r") as f:
    data = json.load(f)
ft_data_index = [item["index"] for item in data if item.get("source") in ["ft"]]
harmful_data_index = [item["index"] for item in data if item.get("source") in ["harmful"]]
try:
    plt.style.use("robox.mplstyle")
except:
    pass

# Set Chinese font for proper display of Chinese characters
plt.rcParams["axes.unicode_minus"] = False

# Customize style
plt.rcParams["xtick.major.size"] = 5
plt.rcParams["xtick.minor.size"] = 2

# Read .pt file data and create dataset
data = {}
all_data={}
for i in range(1, 21):  # Assume there are 20 .pt files
    if i<=5:
        continue
    if "scalar" in path:
        tensor = torch.load(f"{path}/p_steps/scalar_{transformation}_step{i*step}.pt")
    elif "neural" in path:
        tensor = torch.load(f"{path}/p_steps/neural_{transformation}_step{i*step}.pt")
    tensor=tensor.flatten()
    all_tensor=tensor.flatten()
    if tensor.ndim > 1:
        tensor = tensor.flatten()  # Flatten multi-dimensional Tensor
    if flag=="ft":
        tensor=tensor[ft_data_index]
    elif flag=="harmful":
        tensor=tensor[harmful_data_index]
    elif flag=="all":
        tensor=tensor
    else:
        raise NotImplementedError
    data[f"{i * step}"] = tensor.detach().cpu().numpy()  # Convert to NumPy array and store in dictionary
    all_data[f"{i * step}"] = all_tensor.detach().cpu().numpy()
# sorted_data = dict(sorted(data.items(), key=lambda x: int(x[0].split()[-1]),reverse=True))
sorted_data = data
print(sorted_data.keys())


# Configurable parameters
step_z = 0.3  # Control spacing from top to bottom
# Extract maximum and minimum values from all arrays
all_values = [value for values in all_data.values() for value in values]
max_value = max(all_values)
min_value = min(all_values)
print("max",max_value,"min",min_value)
x_limit = [min_value, max_value]  # x-axis range
# x_limit=[0,2]
colors = ["#8E0F31", "#ffffff", "#024163"]  # Set gradient fill colors
cmap = LinearSegmentedColormap.from_list("custom_colors", colors)
step_size = 10  # Set step size to control gradient fill gradient

# Plot graph
fig, ax = plt.subplots(figsize=(12, 12), dpi=150)  # Adjust canvas size for top-to-bottom plotting

# Plot each group of data
for i, (label, values) in enumerate(sorted_data.items()):
    # Calculate density estimation for each group of data
    # values=torch.sigmoid(torch.tensor(values))
    values=(torch.tensor(values))
    if flag=="harmful":
        kde = gaussian_kde(values,0.5)#0.01 for all
    else:
        kde = gaussian_kde(values,0.01)#0.01 for all
    x = np.linspace(*x_limit, 1000)
    y = kde(x)

    # Use gradient fill
    z = -(i+1) * step_z 
    norm = plt.Normalize(x.min(), x.max())
    for j in range(0, len(x) - 1, step_size):
        ax.fill_between(
            x[j : j + step_size + 1],
            y[j : j + step_size + 1] + z,
            z,
            color=cmap(norm(x[j])),
            zorder=i + 1,  
        )

    # Draw curve
    ax.plot(x, y + z, color="#555", linewidth=0.6,zorder=i + 1)

# Set Y-axis labels
steps = list(sorted_data.keys())
yticks = [-((i+1) * step_z) for i in range(len(sorted_data))]
ax.set_yticks(yticks)
ax.set_yticklabels(steps, fontsize=20)

# Decorate plot details
ax.set_xlim(*x_limit)  # Set X-axis range
ax.tick_params(axis='x', labelsize=30)  # Set X-axis tick font size
ax.set_xlabel("")  # Clear X-axis label
ax.set_ylabel("")  # Clear Y-axis label
# ax.set_title(")", fontsize=16, x=0.5, y=1.02)
# Hide all borders
for spine in ax.spines.values():
    spine.set_visible(False)
plt.tight_layout()

# Display and save graph
plt.savefig(f"{path}/p_steps/all_{flag}.png")
plt.show()



