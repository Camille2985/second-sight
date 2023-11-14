import pickle
from src.logging import Logger
from src.preprocessing import load_data, split_data, save_model, data_preprocessing
from src.postprocessing import evaluate
import pandas as pd
from src.torch_processing import IAMDataset
from transformers import TrOCRProcessor


def pipeline():
    output_path = "output/evaluation"

    # Step 1 - Create Logger
    logger = Logger(output_path)
    logger.log("Pipeline started", False)

    # Step 2 - Load the font-recognition-data
    words_df = pd.read_csv("data/labels.csv")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-stage1")
    dataset = IAMDataset(df=words_df, processor=processor, max_target_length=20)

    # Step 5 - Load a model 
    logger.log("Loading model", False)
    model = pickle.load(open("output/model-large-epoch-20.pkl", 'rb'))
    
    # Step 6 - Evaluate the model
    logger.log("Evaluating model", False)
    results = evaluate(model, dataset, 1, logger) 

    # Step 7 - Record the results
    logger.log(results)
    print(results)


if __name__ == "__main__":
    pipeline()
