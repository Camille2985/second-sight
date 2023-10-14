from src.logging import Logger
from src.model_training import fine_tune
from src.preprocessing import load_data, split_data, save_model, data_preprocessing
from datetime import datetime

def pipeline():
    data_path = "data"
    output_path = "output/logs"

    # Step 1 - Create Logger
    logger = Logger(f"{output_path}-{datetime.now()}.txt")
    logger.log("---Pipeline started---")

    # Step 2 - Import the data
    logger.log("---Loading data---")
    words_list = load_data(data_path, logger)

    # Step 3 - Split the data into train and test sets
    logger.log("---Splitting data---")
    train, validation, test = split_data(words_list, logger)

    # Step 4 - Data Prep
    logger.log("---Preprocessing data---")
    train_dataset, validation_dataset, test_dataset = data_preprocessing(data_path, train, validation, test, logger)

    # Step 5 - Train the model
    logger.log("---Training model---")
    tuned_regressor = fine_tune(train_dataset, validation_dataset, epochs=50)

    # Step 6 - Save the model artifacts
    logger.log("---Saving model---")
    save_model(tuned_regressor, "./model_training/output/model.pkl")

if __name__== "__main__":
    pipeline()
