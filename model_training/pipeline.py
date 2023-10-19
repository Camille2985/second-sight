from src.logging import Logger
from src.model_training import fine_tune
from src.preprocessing import load_data, split_data, save_model, data_preprocessing


def pipeline():
    working_locally = True
    data_path = "data"
    output_path = "output"
    epochs = 10 if working_locally else 10
    gpu = False if working_locally else True
    steps = 25 if working_locally else 200

    # Step 1 - Create Logger
    logger = Logger(output_path)
    logger.log("Pipeline started", False)

    # Step 2 - Import the data
    logger.log("Loading data", False)
    words_list = load_data(data_path, logger)

    # Step 3 - Split the data into train and test sets
    logger.log("Splitting data", False)
    train, validation, test = split_data(words_list, logger)

    if working_locally:
        train = train[:10000]
        validation = validation[:2000]
        # test = test[:100]

    # Step 4 - Data Prep
    logger.log("Preprocessing data", False)
    train_dataset, validation_dataset, test_dataset = data_preprocessing(data_path, train, validation, test, logger)

    # Step 5 - Train the model
    logger.log("Training model", False)
    fine_tune(epochs, train_dataset, validation_dataset,  output_path, gpu, steps)


if __name__ == "__main__":
    pipeline()
