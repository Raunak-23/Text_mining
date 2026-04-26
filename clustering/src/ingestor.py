import pandas as pd
import requests
import io
import zipfile

class GDELTIngestor:
    def __init__(self):
        self.last_update_url = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"

    def get_latest_gkg_url(self):
        try:
            resp = requests.get(self.last_update_url)
            lines = resp.text.strip().split('\n')
            
            # Specifically look for the GKG file in the update list
            for line in lines:
                if '.gkg.csv.zip' in line.lower():
                    return line.split(' ')[2]
            
            return lines[2].split(' ')[2]
        except Exception as e:
            print(f"Error fetching GDELT update list: {e}")
            return None

    def fetch_data(self):
        url = self.get_latest_gkg_url()
        if not url: return None
        
        print(f"📥 Downloading latest GKG: {url.split('/')[-1]}")
        resp = requests.get(url)
        
        try:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    df = pd.read_csv(f, sep='\t', header=None, low_memory=False, encoding='utf-8')
        except Exception as e:
            return None
        
        # Dynamic column mapping to prevent 'Length Mismatch'
        num_cols = len(df.columns)
        gkg_headers = [
            'GKGRECORDID', 'DATE', 'SourceCollection', 'SourceCommonName',
            'DocumentIdentifier', 'Counts', 'V2Counts', 'Themes', 'V2Themes', 
            'Locations', 'V2Locations', 'Persons', 'V2Persons', 'Organizations', 
            'V2Organizations', 'V2Tone', 'Dates', 'GCAM', 'SharingImage', 
            'RelatedImages', 'SocialImageEmbeds', 'SocialVideoEmbeds', 'Quotations', 
            'AllNames', 'Amounts', 'TranslationInfo', 'ExtrasXML'
        ]

        if num_cols == 27:
            df.columns = gkg_headers
        else:
            # Handle cases where GDELT sends V1 or extra columns
            new_headers = gkg_headers[:num_cols] + [f'Extra_{i}' for i in range(num_cols - len(gkg_headers))]
            df.columns = new_headers[:num_cols]
             
        return df