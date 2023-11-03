from src.logging import Logger
from src.preprocessing import load_data, split_data, clean_data
import pandas as pd


def pipeline():
    predictions = pd.read_csv("camille_evaluation/large-predictions.csv")
    files = pd.read_csv("camille_evaluation/file_locations.csv")
    predictions["location"] = files["location"]
    predictions["actuals"] = predictions["actuals"].str.replace("[", "")
    predictions["actuals"] = predictions["actuals"].str.replace("]", "")
    predictions["actuals"] = predictions["actuals"].str.replace("'", "")
    predictions["predictions"] = predictions["predictions"].str.replace("[", "")
    predictions["predictions"] = predictions["predictions"].str.replace("]", "")
    predictions["predictions"] = predictions["predictions"].str.replace("'", "")
    predictions.to_csv("camille_evaluation/large-results.csv", index=False)
    print(predictions)


if __name__ == "__main__":
    pipeline()
