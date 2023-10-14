
import numpy as np
import os
from transformers import VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, \
    TrOCRProcessor, TrainerCallback, default_data_collator
import torch

def fine_tune(epochs, train_dataset, validation_dataset):
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        fp16=True,  # must be connected to a GPU for this line to work
        output_dir="../output",
        logging_steps=2,
        save_steps=1000,
        eval_steps=200,
        num_train_epochs=epochs
    )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=eval_metrics,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=default_data_collator,
        callbacks=[SaveEvaluationResultsCallback(processor)]
    )
    trainer.train()
    return trainer

def accuracy_by_letter(pred, actual):
    max_length = max(len(pred), len(actual))
    min_length = min(len(pred), len(actual))
    letters_correct = 0
    for i in range(min_length):
        if pred[i] == actual[i]:
            letters_correct += 1
    return letters_correct / max_length


def eval_metrics(pred, processor):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    # cer
    cer_metric = load_metric("cer")
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    # letter accuracy, word accuracy
    acc_letter = []
    acc_word = 0
    for i in range(len(pred_str)):
      acc_letter.append(accuracy_by_letter(pred_str[i], label_str[i]))
      if pred_str[i] == label_str[i]:
        acc_word += 1

    print("hello Camille!!!")

    return {"cer": cer, "letter_accuracy": np.mean(acc_letter), "word_accuracy": acc_word/len(label_str)}



class SaveEvaluationResultsCallback(TrainerCallback):

    def on_epoch_end(self, args, state, control, model, tokenizer, data_collator=None, compute_metrics=None, **kwargs):
        output_dir = args.output_dir
        epoch = state.epoch
        total_epochs = args.num_train_epochs  # Get the total number of training epochs from args

        if epoch == total_epochs:
          # this needs to be replaced with a reference to our eval_metrics function above,
          # but I could't figure out how to do it
          # right now it's just calculating random numbers
          evaluation_metrics = {
            "metric1": torch.rand(1).item(),
            "metric2": torch.rand(1).item(),
          }

          with open(os.path.join(output_dir, f"evaluation_results_epoch{epoch}.txt"), "w") as file:
              for metric, value in evaluation_metrics.items():
                  file.write(f"{metric}: {value}\n")
