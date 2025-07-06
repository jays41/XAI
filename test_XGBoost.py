import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from fetch_parse_data import fetch_parse_data
from XGBoost import *

def parse_data(df, threshold = 1, n = 1):
    df["n_day_return"] = np.zeros(len(df))
    for i in range(len(df)):
        if i + n >= len(df):
            df.iloc[i, df.columns.get_loc("n_day_return")] = -2
            continue
        cur = df.iloc[i]["Close"]
        change = 100 * (df.iloc[i+n]["Close"] - cur) / cur
        if change > threshold:
            df.iloc[i, df.columns.get_loc("n_day_return")] = 1
        elif change < -threshold:
            df.iloc[i, df.columns.get_loc("n_day_return")] = -1
    return df[df["n_day_return"] != -2]

df = fetch_parse_data()
df = parse_data(df)

df = df.dropna()
df = df.copy()

# Convert n_day_return from [-1.0, 0.0, 1.0] to class labels [0, 1, 2]
label_map = {-1.0: 0, 0.0: 1, 1.0: 2}
df['label'] = df['n_day_return'].map(label_map)

X = df.drop(['n_day_return', 'label'], axis=1).values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multi-class XGBoost training using softmax
n_estimators = 15
learning_rate = 0.05
n_classes = 3

print(f"Training data shape: {X_train.shape}")
print(f"Class distribution in training: {np.bincount(y_train)}")

X_train_list = X_train.tolist()
X_test_list = X_test.tolist()
y_train_list = y_train.tolist()


n_samples = len(X_train_list)
current_predictions_matrix = [[0.0 for _ in range(n_classes)] for _ in range(n_samples)]

trees_by_class = {0: [], 1: [], 2: []}

print(f"\nTraining multi-class XGBoost with {n_estimators} rounds, learning rate {learning_rate}")

for round_idx in range(n_estimators):
    print(f"Training round {round_idx + 1}/{n_estimators}")
    
    # Train one tree per class
    for class_idx in range(n_classes):
        tree = train_one_tree_multiclass(
            X_train_list, y_train_list, current_predictions_matrix, class_idx,
            max_depth=4, min_samples_split=3, l2_reg=1.0, gamma=0.1
        )
        trees_by_class[class_idx].append(tree)
        
        outputs = predict(tree, X_train_list)
        current_predictions_matrix = update_predictions_multiclass(
            current_predictions_matrix, outputs, class_idx, learning_rate
        )

final_probs = predict_multiclass_probs(trees_by_class, X_test_list, learning_rate, n_classes)
final_preds = predict_multiclass_labels(trees_by_class, X_test_list, learning_rate, n_classes)


print("MULTI-CLASS XGBOOST RESULTS")
print("-" * 30)
print(f"Test data shape: {X_test.shape}")
print(f"Class distribution in test: {np.bincount(y_test)}")
print()

accuracy = accuracy_score(y_test, final_preds)
precision = precision_score(y_test, final_preds, average='macro')
recall = recall_score(y_test, final_preds, average='macro')
f1 = f1_score(y_test, final_preds, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"F1 Score (macro): {f1:.4f}")
print()

print("Confusion Matrix:")
cm = confusion_matrix(y_test, final_preds)
print(cm)
print()

class_names = ['Down (-1)', 'Neutral (0)', 'Up (+1)']
print("Class-wise Performance:")
for i in range(3):
    class_precision = precision_score(y_test, final_preds, labels=[i], average=None)
    class_recall = recall_score(y_test, final_preds, labels=[i], average=None)
    class_f1 = f1_score(y_test, final_preds, labels=[i], average=None)
    if len(class_precision) > 0:
        print(f"{class_names[i]}: Precision={class_precision[0]:.3f}, Recall={class_recall[0]:.3f}, F1={class_f1[0]:.3f}")

print()

print("TRADING SIMULATION:")
print("-" * 30)
correct_predictions = np.sum(y_test == final_preds)
total_predictions = len(y_test)
print(f"Correct predictions: {correct_predictions}/{total_predictions} ({100*correct_predictions/total_predictions:.1f}%)")

high_conf_threshold = 0.6
high_conf_mask = np.max(final_probs, axis=1) > high_conf_threshold
high_conf_preds = np.array(final_preds)[high_conf_mask]
high_conf_true = np.array(y_test)[high_conf_mask]

if len(high_conf_preds) > 0:
    high_conf_accuracy = accuracy_score(high_conf_true, high_conf_preds)
    print(f"High-confidence predictions (>{high_conf_threshold:.1f}): {len(high_conf_preds)}/{total_predictions} ({100*len(high_conf_preds)/total_predictions:.1f}%)")
    print(f"High-confidence accuracy: {high_conf_accuracy:.4f} ({100*high_conf_accuracy:.1f}%)")
else:
    print("No high-confidence predictions found")