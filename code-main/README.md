# AICOMP Automated Essay Scoring 2.0

Welcome to our repository for **AICOMP Automated Essay Scoring 2.0**, developed for the [Kaggle Challenge: Automated Essay Scoring 2.0](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/overview).

The competition aims to advance the field of automated grading, leveraging modern techniques to provide timely feedback for student essays, particularly in underserved communities. Your contributions and innovations can make a real impact in reducing the expense and time required for manual grading.

---

## Resources
- [HuggingFace Models and Data](https://huggingface.co/HSLU-AICOMP-LearningAgencyLab)
- [Weights & Biases Project](https://wandb.ai/hslu_nlp/HSLU-AICOMP-LearningAgencyLab)
- [Documentation](https://gitlab.switch.ch/hslu/edu/bachelor-computer-science/aicomp/student-repositories/hs24/automated-essay-scoring-2.0/documentation)

---

## Repository Structure

### Directories
- **`notebooks/`**
  - Jupyter Notebooks, organized into topic-specific subfolders.

- **`scripts/`**
  - Python scripts for dataset creation, Weights & Biases (W&B) sweeps, and other utilities.

- **`tests/`**
  - Python tests.

- **`src/`**
  - Contains minimal local data and serves as the main source folder. All models, trainers, and datasets are stored on [HuggingFace](https://huggingface.co/HSLU-AICOMP-LearningAgencyLab).
  - Key submodules:
    - **`evaluation/`**: Tools and scripts for evaluating model performance.
    - **`models/`**: Stores models pulled from HuggingFace (note: these models are not pushed to the GitLab repository).

---

## Getting Started

### Step 1: Set Up the Development Environment

#### Create and activate the Conda environment:
```bash
conda create --name "aicomp" python=3.11.9
conda activate aicomp
pip install -r requirements.txt

# Install pre-commit hook for linting and formatting
pre-commit install
```

#### Run all linters on all files:
```bash
pre-commit run --all-files
```

### Step 2: Configure Environment Variables
1. Create a `.env` file based on the provided `example.env` file.
2. Add your Kaggle username and API key to the `.env` file.

---

## Code and Testing Conventions

### Linting and Formatting
- **Code style**: `black`
- **Import sorting**: `isort`
- **Docstring checking**: `darglint`
  - *Docstring style*: `sphinx`

### Testing and Notebook Cleanup
- **Tests**: `pytest`
- **Notebook cleanup**: `nbclean` (cleans Jupyter Notebooks before committing).

### Branch Management
- The `main` branch is protected and requires a merge request with approval for any changes.

---

## Team
- **Jannik Bundeli**
- **Jannine Meier**
- **Leon Krug**


### Coach
- **Elena Nazarenko**

---

## Kaggle Challenge Overview

This challenge revisits automated essay scoring with an updated dataset and modern advancements in AI. The competition aims to develop reliable, automated grading techniques to reduce the cost and time associated with hand grading essays. By doing so, it seeks to enable widespread use of essay-based assessments as a key indicator of student learning.

---
