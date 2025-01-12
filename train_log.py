import matplotlib.pyplot as plt

def plot_curve(train_losses, train_reco_losses, train_kl_losses, val_losses, val_reco_losses, val_kl_losses, epochs):
    plt.figure(figsize=(15,5))
    
    # Total loss curve
    plt.subplot(1,3,1)
    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Total Loss Curve")
    plt.legend()
    
    # Reconstruction loss curve
    plt.subplot(1,3,2)
    plt.plot(range(1, epochs + 1), train_reco_losses, label="Training Reconstruction Loss")
    plt.plot(range(1, epochs + 1), val_reco_losses, label="Validation Reconstruction Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss Curve")
    plt.legend()
    
    # KL divergence loss curve
    plt.subplot(1,3,3)
    plt.plot(range(1, epochs + 1), train_kl_losses, label="Training KL Loss")
    plt.plot(range(1, epochs + 1), val_kl_losses, label="Validation KL Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("KL Divergence Loss Curve")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("/home/gou/Programs/fish/result/learning_curve.jpeg")
    print("Learning curves have been saved")
    plt.show()


