import numpy as np
import pandas as pd
def predict(model, x_test, y_test):
    prediction = model.predict(x_test)
    results = []
    for i in range(len(x_test)):
        predicted_bad = prediction[i][0]
        predicted_good = prediction[i][1]
        actual_bad = y_test[i][0]
        actual_good = y_test[i][1]
        results.append({"Sample": i+1,
                        "Predicted_bad_label": predicted_bad, "Actual_bad_label": actual_bad,
                        "predicted_good_label":predicted_good, "Actual_good_label":actual_good})
        print(f"Sample {i+1} - Predicted_bad_label: {predicted_bad:.3f}, Predicted_good_label: {predicted_good:.3f},"
              f"Actual_bad_label: {actual_bad}, Actual_good_label: {actual_good}")
    df = pd.DataFrame(results)
    df.to_csv("/home/gou/Programs/fish/result/Predicted.csv", index = False)
    print("predict result has been saved")
