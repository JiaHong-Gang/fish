import matplotlib.pyplot as plt
import pandas as pd
import os
def training_log(history):
    file_name = "/home/gang/fish/programs/result/train_log.csv"
    os.makedirs(os.path.dirname(file_name), exist_ok = True)
    df = pd.DataFrame(
        {
            "total loss":history.history.get("loss"),
            "reconstruction loss":history.history.get("reconstruction_loss"),
            "kl loss":history.history.get("kl_loss"),
            "val total loss":history.history.get("val_loss"),
            "val reconstruction loss":history.history.get("val_reconstruction_loss"),
            "val kl loss":history.history.get("val_kl_loss")
        }
    )
    df.to_csv(file_name, index_label = "epoch")
    print(f"train log has been saved to :{file_name}")
def learning_curve(history):
    epochs = range(1, len(history.history["loss"]) + 1)
    #total loss curve
    plt.figure(figsize = (12,4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history.history["loss"], label = "total train loss")
    plt.plot(epochs, history.history["val_loss"], label = "total validation train loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("total loss")
    plt.legend()

    #reconstruction loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history.history["reconstruction_loss"], label = "train reconstruction loss")
    plt.plot(epochs, history.history["val_reconstruction_loss"], label = "validation reconstruction loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("reconstruction loss")
    plt.legend()

    #KL loss
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history.history["kl_loss"], label = "train kl loss")
    plt.plot(epochs, history.history["val_kl_loss"], label = "validation kl loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("kl loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("/home/gang/programs/fish/result/learning_curve.jpeg")
    print("learning result has been saved")