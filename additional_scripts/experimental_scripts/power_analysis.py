import pandas as pd
import numpy as np
from statsmodels.stats.power import TTestIndPower

# Load the results
df = pd.read_csv('classification_results_protein_kinase_activity_gpt4o-1.tsv', sep='\t')

# Define a function to calculate power
def calculate_power(effect_size, nobs, alpha=0.05):
    power_analysis = TTestIndPower()
    power = power_analysis.solve_power(effect_size=effect_size, nobs1=nobs, alpha=alpha, ratio=1.0, alternative='two-sided')
    return power

# Calculate metrics
true_positives = df[df['classification'] == 'true_positive'].shape[0]
true_negatives = df[df['classification'] == 'true_negative'].shape[0]
false_positives = df[df['classification'] == 'false_positive'].shape[0]
false_negatives = df[df['classification'] == 'false_negative'].shape[0]

total = true_positives + true_negatives + false_positives + false_negatives

accuracy = (true_positives + true_negatives) / total
precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

# Assume an effect size
effect_size = 0.5  # Medium effect size as per Cohen's conventions

# Calculate power
power_analysis = TTestIndPower()
power = calculate_power(effect_size, total)

# Determine the required sample size for a power of 0.8
required_sample_size = power_analysis.solve_power(effect_size=effect_size, power=0.8, alpha=0.05, ratio=1.0, alternative='two-sided')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")
print(f"Total Samples: {total}")
print(f"Calculated Power: {power:.2f}")
print(f"Required Sample Size for 80% Power: {required_sample_size:.0f}")
