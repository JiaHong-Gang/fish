from train import train_model
from trainlog import plot_curve
from predict import predict
def main():
    model, history, x_val, y_val = train_model()
    plot_curve(history)
if __name__ == '__main__':
    main()

