
class extract_bin:
    def __init__(self, file_path = "sentiment labelled sentences/amazon_cells_labelled.txt"):

        # Initialize lists for inputs and labels
        inputs = []
        labels = []

        # Mapping sentiment to labels
        sentiment_map = {
            "Positive": 1,
            "Neutral": 0.5,
            "Negative": 0
        }
        line_ctr = 0
        lines_list = []
        # Read the file and process each line
        with open(file_path, "r") as file:
            for line in file:
                line_ctr = line_ctr + 1
                line = line.strip()  # Remove any leading/trailing whitespace
                if not line:
                    continue  # Skip empty lines
                
                # Split the sentence and sentiment
                parts = line.rsplit(' ', 1)  # Split at the last space
                
                try:
                    sentence, sentiment = parts[0], parts[1]
                except:
                    sentence = parts[0]
                    sentiment = -1
                    lines_list.append(line_ctr)
                
                # Append the sentence and corresponding label
                inputs.append(sentence)
                labels.append(sentiment)

        self.inputs = inputs
        self.labels = labels
        self.lines_list = lines_list
