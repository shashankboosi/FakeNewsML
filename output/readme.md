# 9417_MachineLearning

Assignment - Fake news

## Output

The purpose of the output folder is to save your output csvs into.
The output csvs are then fed into 'scorer.py' using:

`python scorer.py <test file> <model predictions file>`

Each csv should have a 3-column header: Headline, Body ID, Stance. Each following row is a headline, body ID and true or predicted stance

# Scoring is as follows:
`  +0.25 for each correct unrelated
  +0.25 for each correct related (label is any of agree, disagree, discuss)
  +0.75 for each correct agree, disagree, discuss`