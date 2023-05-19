import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load your datasets
train_original = pd.read_csv("../data/training_set_binary_dif.csv")
test_original = pd.read_csv("../data/test_set_binary_dif.csv")
train_norm = pd.read_csv("../data/training_set_binary_dif_norm.csv")
test_norm = pd.read_csv("../data/test_set_binary_dif_norm.csv")
train_stand = pd.read_csv("../data/training_set_binary_dif_stand.csv")
test_stand = pd.read_csv("../data/test_set_binary_dif_stand.csv")
train_norm_full = pd.read_csv("../data/training_set_binary_dif_norm_full.csv")
test_norm_full = pd.read_csv("../data/test_set_binary_dif_norm_full.csv")

# Concatenate the train and test datasets
dataset_original = pd.concat([train_original, test_original]).drop("class", axis=1)
dataset_norm = pd.concat([train_norm, test_norm]).drop("class", axis=1)
dataset_stand = pd.concat([train_stand, test_stand]).drop("class", axis=1)
dataset_norm_full = pd.concat([train_norm_full, test_norm_full]).drop("class", axis=1)

dataset = dataset_original
label = "Original Dataset 1078"
print(f"\n\n**{label}**\n")
# Compute sparsity
sparsity_before = 1 - (np.count_nonzero(dataset.to_numpy()) / dataset.size)
print(f"Sparsity: {sparsity_before:.2%}")

# Compute the sum of drugs received by each patient
drug_sums_per_patient = dataset.sum(axis=1)

# Calculate the minimum, maximum, and average number of drugs received
min_drugs = drug_sums_per_patient.min()
max_drugs = drug_sums_per_patient.max()
avg_drugs = drug_sums_per_patient.mean()
print(f'\nMinimum number of drugs received by a patient: {min_drugs}')
print(f'Maximum number of drugs received by a patient: {max_drugs}')
print(f'Average number of drugs received by a patient: {avg_drugs:.2f}')

drug_counts = dataset.sum()
min_frequency = drug_counts.min()
max_frequency = drug_counts.max()
avg_frequency = drug_counts.mean()
print(f'\nMinimum frequency of a drug in the dataset: {min_frequency}')
print(f'Maximum frequency of a drug in the dataset: {max_frequency}')
print(f'Average frequency of a drug in the dataset: {avg_frequency:.2f}')

plt.figure(figsize=(10, 6))
sns.histplot(drug_counts, bins=50, kde=False)
plt.xlabel('Frequency')
plt.ylabel('Number of Drugs')
plt.title(f'Histogram of Drug Frequencies ({label})')
plt.show()

