import matplotlib.pyplot as plt
import pandas as pd
import os
#save train log in csv file
def save_train_log(history):
    filename = "/home/gang/programs/fish/result/log_history.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df = pd.DataFrame(
        {
            "train_loss":history.history.get("loss"),
            "validation_loss":history.history.get("val_loss"),
            "train_iou_metric":history.history.get("iou_metric"),
            "validation_iou_metric":history.history.get("val_iou_metric"),
        }
    )
    df.to_csv(filename, index_label = "epoch")
    print("training log has been saved to:", filename)
#draw learning curve
def plot_curve(history):
    plt.figure(figsize=(15,5))
    
    # Total loss curve
    plt.subplot(1,1,1)
    plt.plot(history.history["loss"],label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("/home/gang/programs/fish/result/learning_curve.jpeg")
    print("Learning curves have been saved")
    plt.show()


