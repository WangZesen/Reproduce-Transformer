import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend import Legend

adam_log_dirs = [
    "finals/4772834",
    "finals/4772835",
    "finals/4772836"
]

d_accumadam_ac_log_dirs = [
    "finals/4772411",
    "finals/4772410",
    "finals/4772409"
]

d_accumadam_log_dirs = [
    "finals/4772397",
    "finals/4772398",
    "finals/4771852"
]

def load_log(log_dir, label):
    log = pd.read_csv(os.path.join(log_dir, "test_log.csv"))
    log["label"] = label
    return log

data = pd.DataFrame()

for log_dir in adam_log_dirs:
    log = load_log(log_dir, "Adam")
    data = pd.concat([data, log], ignore_index=True)

for log_dir in d_accumadam_log_dirs:
    log = load_log(log_dir, "DAdam")
    data = pd.concat([data, log], ignore_index=True)

for log_dir in d_accumadam_ac_log_dirs:
    log = load_log(log_dir, "DAdam-AC (ours)")
    data = pd.concat([data, log], ignore_index=True)

for label in ["Adam", "DAdam", "DAdam-AC (ours)"]:
    print(label)
    print(data[(data["label"] == label) & (data["epoch"] == data["epoch"].max())]["BLEU Score"].mean())
    print(data[(data["label"] == label) & (data["epoch"] == data["epoch"].max())]["BLEU Score"].std())
    print(data[(data["label"] == label) & (data["epoch"] == data["epoch"].max())]["train_loss"].mean())
    print(data[(data["label"] == label) & (data["epoch"] == data["epoch"].max())]["train_loss"].std())
    print(data[(data["label"] == label) & (data["epoch"] == data["epoch"].max())]["val_loss"].mean())
    print(data[(data["label"] == label) & (data["epoch"] == data["epoch"].max())]["val_loss"].std())
    print()


sns.set_theme(style="whitegrid")
plt.figure(figsize=(6, 4))
sns.lineplot(data=data, x="epoch", y="BLEU Score", hue="label", errorbar=("ci", 95))
plt.ylim(bottom=0.24)
plt.xlabel("Epoch")
plt.ylabel("BLEU Score")
plt.title("BLEU score over epochs")
plt.legend(title="Algorithm")
plt.xlim(0, 20)
# show only interger x-ticks
plt.xticks(range(0, 21, 2))

plt.tight_layout()
plt.savefig("bleu.png", dpi=300)
plt.savefig("bleu.svg")
plt.clf()

sns.set_theme(style="whitegrid")
plt.figure(figsize=(6, 4))
sns.lineplot(data=data, x="epoch", y="train_loss", hue="label", errorbar=("ci", 95), style=None)
sns.lineplot(data=data, x="epoch", y="val_loss", hue="label", errorbar=("ci", 95), linestyle="--", legend=False)
plt.legend()



combined_handles = []
combined_labels = []
for alg in ["Adam", "DAdam", "DAdam-AC (ours)"]:
    combined_handles.append(Line2D([0], [0], color=sns.color_palette()[{"Adam": 0, "DAdam": 1, "DAdam-AC (ours)": 2}[alg]], linestyle='-'))
    combined_labels.append(alg)

combined_handles.append(Line2D([], [], linestyle=''))
combined_labels.append('')

# Add loss type handles and labels
combined_handles.append(Line2D([0], [0], color='black', lw=2, linestyle='-'))
combined_labels.append('Train Loss')
combined_handles.append(Line2D([0], [0], color='black', lw=2, linestyle='--'))
combined_labels.append('Test Loss')

plt.legend(combined_handles, combined_labels, loc='upper right', title='Algorithm & Loss Type')

plt.ylim(top=3.5)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and test loss over epochs")
plt.xlim(0, 20)
# show only interger x-ticks
plt.xticks(range(0, 21, 2))

plt.tight_layout()
plt.savefig("loss.png", dpi=300)
plt.savefig("loss.svg")
plt.clf()
