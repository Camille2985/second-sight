from src.logging import Logger
from src.model_training import fine_tune
from src.preprocessing import load_data, split_data, save_model, data_preprocessing


def pipeline():
    data_path = "data"
    output_path = "output/logs"
    epochs = 10

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
    from transformers import VisionEncoderDecoderModel
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    # TODO: generate predictions and evaluate


if __name__ == "__main__":
    pipeline()
