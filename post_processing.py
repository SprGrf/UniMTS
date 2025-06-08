# encoding=utf-8
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import pickle

parser = argparse.ArgumentParser(description='argument setting of network')
parser.add_argument('--all_datasets', type=str, nargs='+', default=['C24', 'selfBACK', 'PAMAP2', 'GOTOV', 'DSA', 'MHEALTH', 'HHAR'], 
                    choices=['C24', 'selfBACK', 'PAMAP2', 'GOTOV', 'DSA', 'MHEALTH', 'HHAR'],
                    help='list of dataset names')

parser.add_argument('--metric', type=str, nargs='+', default=['F1'], 
                    choices=['F1', 'precision', 'recall'],
                    help='The metric to use for the main plot')

parser.add_argument('--results_dir', type=str, default=None, help='results directory')


# create directory for saving and plots
global plot_dir_name
plot_dir_name = 'plot/'
if not os.path.exists(plot_dir_name):
    os.makedirs(plot_dir_name)


def boxplot(data):
    plt.boxplot(data, patch_artist=True)  
    plt.title("Box Plot Example")
    plt.xlabel("Dataset")
    plt.ylabel("Values")
    plt.show()


def remove_labels_from_confusion_matrix(cm, labels_to_omit, print_cm = False):
    if len(labels_to_omit) > 0:
        labels_to_omit_set = set(labels_to_omit)
        indices_to_keep = [i for i in range(cm.shape[0]) if i not in labels_to_omit_set]
        cm_filtered = cm[np.ix_(indices_to_keep, indices_to_keep)]
        
        if print_cm:
            print("before")
            print(cm)
            print("after")
            print(cm_filtered)

        return cm_filtered
    else:
        return cm   


def calculate_multiclass_metrics(cm):
    num_classes = cm.shape[0]
    precision_per_class = np.zeros(num_classes)
    recall_per_class = np.zeros(num_classes)
    f1_score_per_class = np.zeros(num_classes)

    # Calculate metrics for each class
    for i in range(num_classes):
        TP = cm[i, i]  # True Positives for class i
        FP = cm[:, i].sum() - TP  # False Positives for class i
        FN = cm[i, :].sum() - TP  # False Negatives for class i
        TN = cm.sum() - (TP + FP + FN)  # True Negatives for class i
       
        # Precision, Recall, F1 Score for class i
        precision_per_class[i] = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall_per_class[i] = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1_score_per_class[i] = (2 * precision_per_class[i] * recall_per_class[i]) / (precision_per_class[i] + recall_per_class[i]) if (precision_per_class[i] + recall_per_class[i]) != 0 else 0


    support = cm.sum(axis=1)  # Number of true instances for each class

    precision_non_zero = precision_per_class[precision_per_class != 0]
    recall_non_zero = recall_per_class[recall_per_class != 0]
    f1_score_non_zero = f1_score_per_class[f1_score_per_class != 0]

    accuracy = np.trace(cm) / cm.sum()  
    precision_macro = precision_non_zero.mean() if precision_non_zero.size > 0 else 0
    recall_macro = recall_non_zero.mean() if recall_non_zero.size > 0 else 0
    f1_score_macro = f1_score_non_zero.mean() if f1_score_non_zero.size > 0 else 0

    weighted_f1 = sum(f1 * s for f1, s in zip(f1_score_per_class, support)) / support.sum()

    classes_present = np.where(support > 0)[0]  # Indices of classes with non-zero support
    mean_recall_present = recall_per_class[classes_present].mean() if len(classes_present) > 0 else 0

    f1_scores_with_negative = list(f1_score_per_class[classes_present])  # Start with F1 scores for present classes
    for i, row_sum in enumerate(support):
        if row_sum == 0 and cm[:, i].sum() > 0:  # Negative class condition
            f1_scores_with_negative.append(0)
            break

    mean_f1_with_negative = np.mean(f1_scores_with_negative) if f1_scores_with_negative else 0

    return {
        'accuracy': accuracy,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_score_per_class': f1_score_per_class,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_score_macro': f1_score_macro,
        'weighted_f1': weighted_f1,
        'mean_recall_present': mean_recall_present,
        'mean_f1_with_negative': mean_f1_with_negative
    }



def process_metrics(confusion_matrices):

    metrics_list = []
    for conm in confusion_matrices:
        
        metrics = calculate_multiclass_metrics(conm)

        metrics_list.append((metrics['mean_recall_present']))

    return metrics_list
    
def search_and_load_files(parent_directory, target_filenames):
    cms = []
    
    for root, _, files in os.walk(parent_directory):
        for file in files:
            if file in target_filenames:
                file_path = os.path.join(root, file)
                print("loading ", file_path)
                with open(file_path, 'rb') as f:
                    conms = pickle.load(f)
                if 'GOTOV' in file_path:
                    omit_label = [4] # running
                elif 'selfBACK' in file_path:
                    omit_label = [5] # cycling
                elif 'HHAR' in file_path:
                    omit_label = [0, 4] # lying, running 
                else:
                    omit_label = []
                for cm in conms:
                    cm_filtered = remove_labels_from_confusion_matrix(cm, omit_label)
                    cms.append(cm_filtered)       
    return cms

if __name__ == '__main__':
    args = parser.parse_args()
    in2in_models = []
    in2out_models = []
    out2in_models = []
    out2out_models = []
    self_test_models = []

    in2in_cms = search_and_load_files(args.results_dir, "in2in.cms")
    in2in_metrics = process_metrics(in2in_cms)

    in2out_cms = search_and_load_files(args.results_dir, "in2out.cms")
    in2out_metrics = process_metrics(in2out_cms)

    self_test_cms = search_and_load_files(args.results_dir, "self.cms")
    self_test_metrics = process_metrics(self_test_cms)

    out2in_cms = search_and_load_files(args.results_dir, "out2in.cms")
    out2in_metrics = process_metrics(out2in_cms)

    out2out_cms = search_and_load_files(args.results_dir, "out2out.cms")
    out2out_metrics = process_metrics(out2out_cms)

    pickle_dict = {}
    pickle_dict["in2in"] = in2in_metrics
    pickle_dict["self"] = self_test_metrics
    pickle_dict["in2out"] = in2out_metrics
    pickle_dict["out2in"] = out2in_metrics
    pickle_dict["out2out"] = out2out_metrics

    with open("UniMTS" + "_metrics.pkl", "wb") as f:
        pickle.dump(pickle_dict, f)  

    # Organize data
    data = [self_test_metrics, in2in_metrics, in2out_metrics, out2in_metrics, out2out_metrics]
    labels = ["SelfTest", "In2In", "In2Out", "Out2In", "Out2Out"]
    
    
    # Create boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=labels)
    plt.xlabel("Metric Type")
    plt.ylabel("Values")
    plt.title("Multi-boxplot of Metrics")
    plt.grid(True)

    # Show plot
    plt.show()
