import matplotlib.pyplot as plt
import pandas as pd

# Data from section 5.7
data = {
    "Stage": ["Before Balancing", "After Balancing"],
    "Accuracy": [70.2, 74.4],
    "Macro_F1": [0.698, 0.744]
}

df = pd.DataFrame(data)

# Create side-by-side bar chart
plt.figure(figsize=(8, 6))
bar_width = 0.35
x = range(len(df))

plt.bar(x, df["Accuracy"], width=bar_width, label="Accuracy (%)")
plt.bar([p + bar_width for p in x], [f1 * 100 for f1 in df["Macro_F1"]],
        width=bar_width, label="Macro F1 (%)")

plt.title("VocalSense AI — Model Performance Before vs After Balancing", fontsize=14, weight='bold')
plt.xlabel("Training Stage", fontsize=12)
plt.ylabel("Score (%)", fontsize=12)
plt.xticks([p + bar_width/2 for p in x], df["Stage"])
plt.ylim(65, 80)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
