import re

#lower() → normalizes case
# Regex [^a-z\s] → removes everything except letters and spaces
# Regex \s+ → collapses multiple spaces into one
# This is exactly the kind of preprocessing step used in traditional NLP pipelines before modeling.

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, keep only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# Example usage
raw_text = "[ALERT]!!! Go to the... KITCHEN & grab the (Apple)."
print(clean_text(raw_text))
