from sentence_transformers import SentenceTransformer
import pandas as pd

class DataProcessor:
    def __init__(self, theme_filter="BLOCKCHAIN"):
        # Small, fast model for streaming
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.theme_filter = theme_filter

    def process(self, df):
        # Filter for your specific topic
       # Change this line inside the process() function:
        mask = df['Themes'].str.contains("BLOCKCHAIN|ECON_CRYPTO|IPL|TECHNOLOGY", na=False, case=False)
        filtered_df = df[mask].copy()
        
        if filtered_df.empty:
            return None, None

        # Extract a clean title from the URL
        def clean_url_to_title(url):
            try:
                return url.split('/')[-1].replace('-', ' ').replace('.html', '')[:100]
            except:
                return "Unknown Title"

        filtered_df['Title'] = filtered_df['DocumentIdentifier'].apply(clean_url_to_title)
        
        # Narrative context = Title + Metadata Themes
        text_to_embed = (filtered_df['Title'] + " " + filtered_df['Themes'].fillna('')).tolist()
        vectors = self.model.encode(text_to_embed)
        
        return filtered_df, vectors