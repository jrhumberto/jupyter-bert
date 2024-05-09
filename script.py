# !pip install datasets transformers[torch] evaluate
from datasets import load_dataset

dataset = load_dataset("glue", "mrpc")

#-
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", num_labels=2)]
mapped_dataset = dataset.map(lambda x: tokenizer(x["sentence1"], x["sentence2"], batched=True))

#-
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#-
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

#-
from transformers import TrainingArguments
import numpy
train_args = TrainingArguments("test-run-1",  evaluation_strategy="epoch")
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = numpy.argmax(logits, axis =- 1)
    return metric.compute(predictions=predictions, references=labels)



from transformers import Trainer

trainer = Trainer(
    model = model,
    args=train_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics = compute_metrics
)

trainer.train()


#-
import numpy as np
preds = model.predict(mapped_dataset['test'])
logits, labels, _ = preds
predictions = np.argmax(logits, axis=-1)



#-
import evaluate
metric = evaluate.load("glue", "mrpc")


#-
metric.compute(predictions=predictions, references=labels)


#- Não vale
'''
from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")

from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions&predictions.shape, predictions. label_ids.shape)
metric = evaluate.load("glue", "mrpc")
logits, labels, _= predictions
predictions = np.argmax(logits, axis =- 1)
print(metric.compute(predictions=predictions, references=labels))

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis =- 1)
    return metric.compute(predictions=predictions, references=labels)

'''
