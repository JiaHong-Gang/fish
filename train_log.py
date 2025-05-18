import matplotlib.pyplot as plt
import pandas as pd
import os
#save train log in csv file
def save_train_log(train_losses, train_reco_losses, train_kl_losses, train_perceptual_losses, val_losses, val_reco_losses, val_kl_losses ,val_perceptual_losses):
    filename = "/home/gang/programs/fish/result/log_history.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df = pd.DataFrame(
        {
            "train_losses":train_losses,
            "train_reco_losses":train_reco_losses,
            "train_kl_losses":train_kl_losses,
            "train_perceptual_losses":train_perceptual_losses,
            "val_losses":val_losses,
            "val_reco_losses":val_reco_losses,
            "val_kl_losses":val_kl_losses,
            "val_perceptual_losses":val_perceptual_losses,
        }
    )
    df.to_csv(filename, index_label = "epoch")
    print("training log has been saved")
#draw learning curve
def plot_curve(history):
    plt.figure(figsize=(15,5))
    
    # Total loss curve
    plt.subplot(1,2,1)
    plt.plot(history.history["loss"],label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    
    # Reconstruction loss curve
    plt.subplot(1,2,2)
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Accuracy Curve")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("/home/gang/programs/fish/result/learning_curve.jpeg")
    print("Learning curves have been saved")
    plt.show()


