# File path to your text file
file_path = "train_set.txt"

# Initialize lists for inputs and labels
inputs = []
labels = []

# Mapping sentiment to labels
sentiment_map = {
    "Positive": 1,
    "Neutral": 0,
    "Negative": -1
}

# Read the file and process each line
with open(file_path, "r") as file:
    for line in file:
        line = line.strip()  # Remove any leading/trailing whitespace
        if not line:
            continue  # Skip empty lines
        
        # Split the sentence and sentiment
        parts = line.rsplit(' ', 1)  # Split at the last space
        sentence, sentiment = parts[0], parts[1]
        
        # Append the sentence and corresponding label
        inputs.append(sentence)
        labels.append(sentiment_map.get(sentiment, None))

# Check the results
print("Inputs:", inputs)
print("Labels:", labels)
