from train import train_model
from picture import plot_curve
from predict import predict
def main():
    model, history, x_test, y_test = train_model()
    plot_curve(history)
    predict(model, x_test, y_test)
if __name__ == '__main__':
    main()

