import math
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def softmax_matrix(logits_matrix):
    result = []
    for logits in logits_matrix:
        result.append(softmax(logits))
    return np.array(result)

def cross_entropy_loss(y_true_class, y_pred_logits):
    probs = softmax(y_pred_logits)
    return -math.log(probs[y_true_class] + 1e-15)  # small epsilon for numerical stability

def cross_entropy_loss_batch(y_true_list, predictions_matrix):
    total_loss = 0.0
    n_samples = len(y_true_list)
    
    for i, true_class in enumerate(y_true_list):
        probs = softmax(predictions_matrix[i])
        sample_loss = -math.log(probs[true_class] + 1e-15)
        total_loss += sample_loss
    
    return total_loss / n_samples

def loss(y_true, y_pred):
    p = sigmoid(y_pred)
    loss = -(y_true * math.log(p) + (1 - y_true) * math.log(1 - p))
    return loss

def leaf_score(gradients, hessians, reg_param):
    return -sum(gradients) / (sum(hessians) + reg_param)

def best_split_gain(g_left, h_left, g_right, h_right, l2_reg, cost):
    g_total = g_left + g_right
    h_total = h_left + h_right
    gain = 0.5 * (
        (g_left ** 2) / (h_left + l2_reg) +
        (g_right ** 2) / (h_right + l2_reg) -
        (g_total ** 2) / (h_total + l2_reg)
    ) - cost
    return gain

def update_prediction(prev_prediction, learning_rate, new_prediction):
    return prev_prediction + learning_rate * new_prediction

def compute_gradients_and_hessians(y, predictions):
    gradients = []
    hessians = []
    for actual_label, prediction in zip(y, predictions):
        predicted_probability = sigmoid(prediction)
        gradients.append(predicted_probability - actual_label)
        hessians.append(predicted_probability * (1 - predicted_probability))
    return gradients, hessians

def compute_gradients_and_hessians_multiclass(y_true, predictions_matrix, class_idx):
    gradients = []
    hessians = []
    
    for i, (true_class, pred_logits) in enumerate(zip(y_true, predictions_matrix)):
        # Convert logits to probabilities using softmax
        probs = softmax(pred_logits)
        prob_k = probs[class_idx]
        
        # Gradient is derivative of cross-entropy w.r.t. logit for class k
        if true_class == class_idx:
            gradient = prob_k - 1  # p_k - 1 when true class is k
        else:
            gradient = prob_k      # p_k when true class is not k
            
        # Hessian is second derivative (for Newton's method)
        hessian = prob_k * (1 - prob_k)
        
        gradients.append(gradient)
        hessians.append(hessian)
    
    return gradients, hessians

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def build_tree(feature_matrix, gradients, hessians, cur_depth, max_depth, min_samples_split, l2_reg, gamma):
    num_samples = len(feature_matrix)
    num_features = len(feature_matrix[0])
    
    # base case
    if cur_depth >= max_depth or num_samples < min_samples_split:
        value = leaf_score(gradients, hessians, l2_reg)
        return Node(value=value)
    
    best_gain = float('-inf')
    best_split_feature = None
    best_split_threshold = None
    best_left_ids = None
    best_right_ids = None
    
    for feature_id in range(num_features):
        unique_thresholds = set(row[feature_id] for row in feature_matrix)
        
        for threshold in unique_thresholds:
            left_ids = [i for i in range(num_samples) if feature_matrix[i][feature_id] <= threshold]
            right_ids = [i for i in range(num_samples) if feature_matrix[i][feature_id] > threshold]
            
            if len(left_ids) < min_samples_split or len(right_ids) < min_samples_split:
                continue
            
            gradient_left = sum(gradients[i] for i in left_ids)
            gradient_right = sum(gradients[i] for i in right_ids)
            hessian_left = sum(hessians[i] for i in left_ids)
            hessian_right = sum(hessians[i] for i in right_ids)
            
            gain = best_split_gain(gradient_left, hessian_left, gradient_right, hessian_right, l2_reg, gamma)
            
            if gain > best_gain:
                best_gain = gain
                best_split_feature = feature_id
                best_split_threshold = threshold
                best_left_ids = left_ids
                best_right_ids = right_ids
    
    if best_gain == float('-inf'):
        value = leaf_score(gradients, hessians, l2_reg)
        # print("No valid split found. Converting to leaf")
        return Node(value=value)

    if best_left_ids is None or best_right_ids is None:
        return Node(value=leaf_score(gradients, hessians, l2_reg))
    
    left_features = [feature_matrix[i] for i in best_left_ids]
    right_features = [feature_matrix[i] for i in best_right_ids]
    left_gradients = [gradients[i] for i in best_left_ids]
    right_gradients = [gradients[i] for i in best_right_ids]
    left_hessians = [hessians[i] for i in best_left_ids]
    right_hessians = [hessians[i] for i in best_right_ids]
    
    left_child = build_tree(left_features, left_gradients, left_hessians, cur_depth + 1, max_depth, min_samples_split, l2_reg, gamma)
    right_child = build_tree(right_features, right_gradients, right_hessians, cur_depth + 1, max_depth, min_samples_split, l2_reg, gamma)
    
    return Node(feature_index=best_split_feature, threshold=best_split_threshold, left=left_child, right=right_child)

def predict_single_sample(node, feature_vector):
    if node.value is not None:
        return node.value
    
    if feature_vector[node.feature_index] <= node.threshold:
        return predict_single_sample(node.left, feature_vector)
    else:
        return predict_single_sample(node.right, feature_vector)

def train_one_tree(feature_matrix, labels, cur_predictions, max_depth=3, min_samples_split=5, l2_reg=1.0, gamma=0.1):
    gradients, hessians = compute_gradients_and_hessians(labels, cur_predictions)
    tree = build_tree(feature_matrix, gradients, hessians, cur_depth=0, max_depth=max_depth, min_samples_split=min_samples_split, l2_reg=l2_reg, gamma=gamma)
    return tree

def predict(tree, feature_matrix):
    return [predict_single_sample(tree, x) for x in feature_matrix]

def update_predictions(prev_predictions, outputs, learning_rate):
    return [prev + learning_rate * update for prev, update in zip(prev_predictions, outputs)]

def train_one_tree_multiclass(feature_matrix, labels, cur_predictions_matrix, class_idx, max_depth=3, min_samples_split=5, l2_reg=1.0, gamma=0.1):
    gradients, hessians = compute_gradients_and_hessians_multiclass(labels, cur_predictions_matrix, class_idx)
    tree = build_tree(feature_matrix, gradients, hessians, cur_depth=0, max_depth=max_depth, min_samples_split=min_samples_split, l2_reg=l2_reg, gamma=gamma)
    return tree

def update_predictions_multiclass(prev_predictions_matrix, tree_outputs, class_idx, learning_rate):
    updated_matrix = [row[:] for row in prev_predictions_matrix]
    for i, output in enumerate(tree_outputs):
        updated_matrix[i][class_idx] += learning_rate * output
    return updated_matrix

def predict_multiclass_probs(trees_by_class, feature_matrix, learning_rate, n_classes=3):
    n_samples = len(feature_matrix)
    # Initialize logits matrix
    logits_matrix = [[0.0 for _ in range(n_classes)] for _ in range(n_samples)]
    
    # Add contributions from all trees
    for class_idx in range(n_classes):
        if class_idx in trees_by_class:
            for tree in trees_by_class[class_idx]:
                outputs = predict(tree, feature_matrix)
                for i, output in enumerate(outputs):
                    logits_matrix[i][class_idx] += learning_rate * output
    
    # Convert logits to probabilities using softmax
    return softmax_matrix(logits_matrix)

def predict_multiclass_labels(trees_by_class, feature_matrix, learning_rate, n_classes=3):
    probs_matrix = predict_multiclass_probs(trees_by_class, feature_matrix, learning_rate, n_classes)
    return [np.argmax(probs) for probs in probs_matrix]