from src.logging import Logger
from src.model_training import fine_tune
from src.preprocessing import load_data, split_data, save_model, data_preprocessing

def pipeline():
    data_path = "data"
    output_path = "output/logs"

    # Step 1 - Create Logger
    logger = Logger(output_path)
    logger.log("Pipeline started", False)

    # Step 2 - Import the data
    logger.log("Loading data", False)
    words_list = load_data(data_path, logger)

    # Step 3 - Split the data into train and test sets
    logger.log("Splitting data", False)
    train, validation, test = split_data(words_list, logger)

    # Step 4 - Data Prep
    logger.log("Preprocessing data", False)
    train_dataset, validation_dataset, test_dataset = data_preprocessing(data_path, train, validation, test, logger)

    # Step 5 - Train the model
    logger.log("Training model", False)
    tuned_regressor = fine_tune(train_dataset, validation_dataset, epochs=50)

    # Step 6 - Save the model artifacts
    logger.log("Saving model", False)
    save_model(tuned_regressor, "./model_training/output/model.pkl")

if __name__== "__main__":
    pipeline()
