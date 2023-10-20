from src.logging import Logger
from src.model_training import fine_tune
from src.preprocessing import load_data, split_data, save_model, data_preprocessing
from src.postprocessing import evaluate
from transformers import VisionEncoderDecoderModel


def pipeline():
    data_path = "data"
    output_path = "output/evaluation"

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

    # Step 5 - Load a model (for now, it's just the pre-trained trOCR, but we'll replace this with our fine-tuned model once it's ready)
    logger.log("Loading model", False)
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten") # replace eventually
    
    # Step 6 - Evaluate the model
    logger.log("Evaluating model", False)
    results = evaluate(model, test_dataset) # note that we use the test set for eval

    # Step 7 - Record the results
    print(results)


if __name__ == "__main__":
    pipeline()
