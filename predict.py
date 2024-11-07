import numpy as np
import pandas as pd
def predict(model, x_test, y_test):
    prediction = model.predict(x_test)
    results = []
    for i in range(len(x_test)):
        predicted_label = prediction[i][1]
        actual_label = y_test[i][1]
        results.append({"Sample": i+1, "Predicted_label": predicted_label, "Actual_label": actual_label})
        print(f"Sample {i+1} - Predicted_label: {predicted_label}, Actual_label: {actual_label}")
    df = pd.DataFrame(results)
    df.to_csv("/home/gou/Programs/fish/result/Predicted.csv", index = False)
    print("predict result has been saved")
