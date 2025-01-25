from dotenv import load_dotenv
load_dotenv()
import os
import pandas as pd
import kaggle as kg
import zipfile
from spellchecker import SpellChecker
from collections import Counter
from datasets import Dataset
import huggingface_hub
import textstat
import string

# Initialize configurations
huggingface_username = 'HSLU-AICOMP-LearningAgencyLab'
competition = 'learning-agency-lab-automated-essay-scoring-2'

# Initialize spell checker
spell = SpellChecker()

# Authenticate with Kaggle API
kg.api.authenticate()

# Download the competition files
kg.api.competition_download_files(competition=competition, path='../src/datasets', quiet=False)

# Extract the zip file
zf = zipfile.ZipFile(f'../src/datasets/{competition}.zip') 

# Load the datasets
submission = pd.read_csv(zf.open('sample_submission.csv'))
test = pd.read_csv(zf.open('test.csv'))
train = pd.read_csv(zf.open('train.csv'))

def remove_punctuation_except_apostrophe(text):
    # Define the punctuation to be removed (exclude apostrophes)
    punctuation_to_remove = string.punctuation.replace("'", "")
    return text.translate(str.maketrans('', '', punctuation_to_remove))


# Function to dynamically compute features
def compute_features(text, spell_checker):
    """
    Compute features for each text input.
    
    Args:
        text (str): The full essay text.
        spell_checker (SpellChecker): A spell checker instance.
    
    Returns:
        dict: A dictionary with dynamically computed features.
    """
    clean_text = remove_punctuation_except_apostrophe(text)
    words = clean_text.split()  # Simple word splitting
    misspelled = spell_checker.unknown(words)  # Identify misspelled words
    # Count how often each misspelled word appears
    misspelled_counts = Counter([word for word in words if word in misspelled])

    # Dynamic feature dictionary
    features = {
        'unique_mistakes': len(misspelled_counts),  # Total unique spelling mistakes
        'repeated_mistakes_count': sum(count for count in misspelled_counts.values() if count > 1),  # Sum of repeated mistakes
        'max_repeated_mistake': max(misspelled_counts.values(), default=0),  # Maximum times a single mistake is repeated,
        'word_count': len(words),
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
    }

    return features

# Function to apply features dynamically to a dataset
def add_features_to_dataset(dataset, spell_checker):
    """
    Adds dynamically computed features to the dataset.

    Args:
        dataset (pd.DataFrame): The input dataset.
        spell_checker (SpellChecker): A spell checker instance.
    
    Returns:
        pd.DataFrame: Dataset with additional features.
    """
    features_df = dataset['full_text'].apply(lambda x: pd.Series(compute_features(x, spell_checker)))
    dataset = pd.concat([dataset, features_df], axis=1)
    return dataset

# Add features to the train and test datasets
train = add_features_to_dataset(train, spell)
test = add_features_to_dataset(test, spell)

# Add a placeholder 'score' column to the test dataset
test['score'] = -1

# Create Hugging Face Datasets from pandas DataFrames
train_dataset = Dataset.from_pandas(train)
test_dataset = Dataset.from_pandas(test)

# Split the train dataset into train and eval (80% train, 20% eval)
train_test_split = train_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Show the updated train dataset
print(train_dataset)

# Show the updated train dataset
print(eval_dataset)

# Show the updated test dataset
print(test_dataset)

# Use your Hugging Face API token to authenticate
huggingface_hub.login(token=os.getenv('HUGGINGFACE_TOKEN'))

# Push the train dataset to Hugging Face as a private dataset with 'train' split
train_dataset.push_to_hub(f"{huggingface_username}/{competition}", private=True, split="train")

# Push the train dataset to Hugging Face as a private dataset with 'train' split
eval_dataset.push_to_hub(f"{huggingface_username}/{competition}", private=True, split="eval")

# Push the test dataset to Hugging Face as a private dataset with 'test' split
test_dataset.push_to_hub(f"{huggingface_username}/{competition}", private=True, split="test")
