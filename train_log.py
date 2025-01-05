from tabnanny import verbose

import matplotlib.pyplot as plt

def plot_curve(history):
    plt.figure(figsize=(12,5))
# loss curve
    plt.subplot(1,2,1)
    plt.plot(history.history["loss"], label = "Training_loss")
    plt.plot(history.history["val_loss"], label = "Validation_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss-curve")
    plt.legend()
    plt.savefig("/home/gou/Programs/fish/result/learning_curve.jpeg")
    print("learning result has been saved")
    plt.show()


