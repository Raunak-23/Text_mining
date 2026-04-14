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
            
            # Specifically look for the GKG file (27 columns)
            for line in lines:
                if '.gkg.csv.zip' in line.lower():
                    return line.split(' ')[2]
            
            # Fallback if the loop fails (GKG is usually index 2)
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
                    # GDELT is tab-separated
                    df = pd.read_csv(f, sep='\t', header=None, low_memory=False, encoding='utf-8')
        except Exception as e:
            print(f"❌ Error unzipping/reading CSV: {e}")
            return None
        
        # GDELT GKG V2 Column Mapping (The 27 columns)
        df.columns = [
            'GKGRECORDID', 'DATE', 'SourceCollection', 'SourceCommonName',
            'DocumentIdentifier', 'Counts', 'V2Counts', 'Themes', 'V2Themes', 
            'Locations', 'V2Locations', 'Persons', 'V2Persons', 'Organizations', 
            'V2Organizations', 'V2Tone', 'Dates', 'GCAM', 'SharingImage', 
            'RelatedImages', 'SocialImageEmbeds', 'SocialVideoEmbeds', 'Quotations', 
            'AllNames', 'Amounts', 'TranslationInfo', 'ExtrasXML'
        ]
        return df