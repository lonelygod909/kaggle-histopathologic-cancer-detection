import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class TrainingLogger:
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.best_val_acc = 0.0
        self.log_dir = log_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_log_{self.timestamp}.json")
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_auc": [], 
            "fold_results": []
        }
    
    def log_epoch(self, epoch, train_loss, train_acc, val_loss=None, val_acc=None, val_auc=None): 
        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc
        }
        
        if val_loss is not None:
            epoch_log["val_loss"] = val_loss
        if val_acc is not None:
            epoch_log["val_acc"] = val_acc
        if val_auc is not None: 
            epoch_log["val_auc"] = val_auc
            
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        
        if val_loss is not None:
            self.history["val_loss"].append(val_loss)
        if val_acc is not None:
            self.history["val_acc"].append(val_acc)
        if val_auc is not None: 
            self.history["val_auc"].append(val_auc)
        
        self._save_log()
        return epoch_log
    
    def log_fold(self, fold, final_train_loss, final_train_acc, final_val_loss, final_val_acc, final_val_auc): 
        fold_result = {
            "fold": fold,
            "final_train_loss": final_train_loss,
            "final_train_acc": final_train_acc,
            "final_val_loss": final_val_loss,
            "final_val_acc": final_val_acc,
            "final_val_auc": final_val_auc
        }
        self.history["fold_results"].append(fold_result)
        self._save_log()
        return fold_result
    
    def log_error(self, error_message):
        error_log = {
            "error": error_message,
            "timestamp": self.timestamp
        }
        self.history["errors"].append(error_log)
        self._save_log()
        return error_log

    def _save_log(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def plot_metrics(self, save=True):
        num_plots = 2
        if len(self.history["val_auc"]) > 0:
            num_plots = 3
        
        plt.figure(figsize=(5 * num_plots, 5))
        
        plt.subplot(1, num_plots, 1)
        plt.plot(self.history["train_loss"], label="Train Loss")
        if len(self.history["val_loss"]) > 0:
            plt.plot(self.history["val_loss"], label="Validation Loss")
        plt.legend()
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        
        plt.subplot(1, num_plots, 2)
        plt.plot(self.history["train_acc"], label="Train Accuracy")
        if len(self.history["val_acc"]) > 0:
            plt.plot(self.history["val_acc"], label="Validation Accuracy")
        plt.legend()
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")

        if num_plots == 3:
            plt.subplot(1, num_plots, 3)
            if len(self.history["val_auc"]) > 0:
                plt.plot(self.history["val_auc"], label="Validation AUC")
            plt.legend()
            plt.title("AUC")
            plt.xlabel("Epoch")
            plt.ylabel("AUC")
        
        plt.tight_layout()
        
        if save:
            plot_file = os.path.join(self.log_dir, f"metrics_plot_{self.timestamp}.png")
            plt.savefig(plot_file)
            
        plt.show()
        
    def summarize_cv_results(self):
        if len(self.history["fold_results"]) == 0:
            return "No cross-validation results available."
        
        train_losses = [fold["final_train_loss"] for fold in self.history["fold_results"]]
        train_accs = [fold["final_train_acc"] for fold in self.history["fold_results"]]
        val_losses = [fold["final_val_loss"] for fold in self.history["fold_results"]]
        val_accs = [fold["final_val_acc"] for fold in self.history["fold_results"]]
        val_aucs = [fold.get("final_val_auc", 0.0) for fold in self.history["fold_results"]] 

        summary = {
            "mean_train_loss": np.mean(train_losses),
            "std_train_loss": np.std(train_losses),
            "mean_train_acc": np.mean(train_accs),
            "std_train_acc": np.std(train_accs),
            "mean_val_loss": np.mean(val_losses),
            "std_val_loss": np.std(val_losses),
            "mean_val_acc": np.mean(val_accs),
            "std_val_acc": np.std(val_accs),
            "mean_val_auc": np.mean(val_aucs), 
            "std_val_auc": np.std(val_aucs)   
        }
        
        self.history["cv_summary"] = summary
        self._save_log()
        
        return summary



