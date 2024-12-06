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
# accuracy curve
    plt.subplot(1,2,2)
    plt.plot(history.history["accuracy"], label = "Training_accuracy")
    plt.plot(history.history["val_accuracy"], label = "Validation_accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy-curve")
    plt.legend()
    plt.savefig("/home/gou/Programs/fish/result/learning_curve.jpeg")
    print("learning result has been saved")
    plt.show()
    """
    test_loss ,test_accuracy, test_iou = model.evaluate(x_test,y_test,verbose = 1)
    print(f"test loss is {test_loss}")
    print(f"test accuracy is {test_accuracy}")
    print(f"test iou is {test_iou}")
    """

