# Opinion Mining System for Analyzing Public Feedback in Smart Governance

## Project Overview
This project is a beginner-friendly Python NLP application that classifies public feedback into three sentiment categories:
- Positive
- Negative
- Neutral

It uses a small sample dataset, text preprocessing, TF-IDF feature extraction, and a Logistic Regression classifier.

## Features
- Loads feedback data from `dataset.csv`
- Preprocesses text using lowercase conversion, punctuation removal, and stopword removal
- Converts text to numerical features with TF-IDF
- Trains a Logistic Regression model
- Prints accuracy and classification report
- Plots a confusion matrix using Matplotlib
- Accepts custom user feedback from the terminal for real-time prediction
- Includes a simple CLI menu

## Installation
1. Open the `opinion-mining-project` folder in VS Code.
2. Create a virtual environment if you want to keep dependencies isolated.
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run
Run the main script:
```bash
python main.py
```

The script will:
1. Load the dataset
2. Preprocess the text
3. Train and evaluate the model
4. Show a confusion matrix
5. Open a simple terminal menu for custom feedback prediction

## Sample Input/Output
### Sample Input
```text
Enter your feedback: The new portal is very helpful and saves time.
```

### Sample Output
```text
Predicted Sentiment: Positive
```

## Dataset
The dataset is stored in `dataset.csv` with the following columns:
- `text`
- `label`

The sample data includes balanced Positive, Negative, and Neutral examples.

## Notes
- The project uses NLTK stopwords and downloads the resource automatically if needed.
- The confusion matrix is saved as `confusion_matrix.png` after execution.
