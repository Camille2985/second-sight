import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_metric
from transformers import TrOCRProcessor
from src.model_training import accuracy_by_letter
import numpy as np
import pandas as pd

def evaluate(model, dataset, batch_size=1, logger=None):
    """Use this function to evaluate the performance of a model on a dataset according to the eval metrics CER, letter_accuracy, and 
    word_accuracy"""

    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")

    logger.log("Running evaluation...")

    acc_letter = []
    acc_word = 0
    cer_metric = load_metric("cer")

    df = pd.DataFrame(columns=["predictions", "actuals"])


    for batch in tqdm(test_dataloader):
        # generate predictions
        pixel_values = batch["pixel_values"].to(device)
        outputs = model.generate(pixel_values)

        # decode predictions
        pred_str = processor.batch_decode(outputs, skip_special_tokens=True)

        # decode labels
        labels = batch["labels"]
        labels[labels == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels, skip_special_tokens=True)

        df = df.append({'predictions': pred_str, 'actuals': label_str}, ignore_index=True)

        # cer
        cer_metric.add_batch(predictions=pred_str, references=label_str)

        # letter accuracy, word accuracy
        for i in range(len(pred_str)):
            acc_letter.append(accuracy_by_letter(pred_str[i], label_str[i]))
            if pred_str[i] == label_str[i]:
                acc_word += 1

    df.to_csv("camille_evaluation/large-predictions.csv", index=False)
    return {"cer": cer_metric.compute(), "letter_accuracy": np.mean(acc_letter), "word_accuracy": acc_word/len(dataset)}
