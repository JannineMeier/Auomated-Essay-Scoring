from dotenv import load_dotenv
load_dotenv()
import os
import huggingface_hub
from datasets import load_dataset
import wandb


# Load dataset from Hugging Face hub
huggingface_username = 'HSLU-AICOMP-LearningAgencyLab'
first_dataset_name = 'learning-agency-lab-automated-essay-scoring-2_V2'
second_dataset_name = 'whole_kaggle_dataset'

new_dataset_name = 'learning-agency-lab-automated-essay-scoring-2_V3'

huggingface_hub.login(token=os.getenv('HUGGINGFACE_TOKEN'))

# Load the dataset from Hugging Face
first_dataset = load_dataset(f"{huggingface_username}/{first_dataset_name}")
second_dataset = load_dataset(f"{huggingface_username}/{second_dataset_name}")

# Inspect dataset before preprocessing
print(first_dataset)
print(second_dataset)

from datasets import DatasetDict

# Add "in_persuade_corpus" from the second dataset to the first dataset by matching "essay_id"
def merge_datasets_with_in_persuade_corpus(first_ds, second_ds):
    second_train = second_ds['train'].to_pandas()
    second_essay_map = second_train.set_index('essay_id')['in_persuade_corpus'].to_dict()

    def add_in_persuade_corpus(example):
        essay_id = example['essay_id']
        if essay_id not in second_essay_map:
            print(f"No match found for essay_id {essay_id}. Setting 'in_persuade_corpus' to False.")
        example['in_persuade_corpus'] = second_essay_map.get(essay_id, False)
        return example

    merged_datasets = DatasetDict({
        split: first_ds[split].map(add_in_persuade_corpus)
        for split in first_ds.keys()
    })

    return merged_datasets

merged_dataset = merge_datasets_with_in_persuade_corpus(first_dataset, second_dataset)

print(merged_dataset)

# Push the train dataset to Hugging Face as a private dataset with 'train' split
merged_dataset['train'].push_to_hub(f"{huggingface_username}/{new_dataset_name}", private=True, split="train")

# Push the train dataset to Hugging Face as a private dataset with 'train' split
merged_dataset['eval'].push_to_hub(f"{huggingface_username}/{new_dataset_name}", private=True, split="eval")

# Push the test dataset to Hugging Face as a private dataset with 'test' split
merged_dataset['test'].push_to_hub(f"{huggingface_username}/{new_dataset_name}", private=True, split="test")