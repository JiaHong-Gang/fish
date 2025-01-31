import matplotlib.pyplot as plt

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
    plt.figure(figsize = (12,4))
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history.history["reconstruction_loss"], label = "train reconstruction loss")
    plt.plot(epochs, history.history["val_reconstruction_loss"], label = "validation reconstruction loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("reconstruction loss")
    plt.legend()

    #KL loss
    plt.figure(figsize = (12,4))
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history.history["kl_loss"], label = "train kl loss")
    plt.plot(epochs, history.history["val_kl_loss"], label = "validation kl loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("kl loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("/home/gou/Programs/fish/result/learning_curve.jpeg")
    print("learning result has been saved")