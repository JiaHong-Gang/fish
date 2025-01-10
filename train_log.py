import matplotlib.pyplot as plt

def plot_curve(train_losses, val_losses, epochs):
    plt.figure(figsize=(12,5))
# loss curve
    plt.subplot(1,2,1)
    plt.plot(range(1,epochs +1), train_losses, label = "Training_loss")
    plt.plot(range(1,epochs +1), val_losses, label = "Validation_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss-curve")
    plt.legend()
    plt.savefig("/home/gou/Programs/fish/result/learning_curve.jpeg")
    print("learning result has been saved")
    plt.show()


