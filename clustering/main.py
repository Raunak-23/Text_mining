import time
import sys
import os
from src.ingestor import GDELTIngestor
from src.processor import DataProcessor
from src.clusterer import IncrementalClusterer
from src.utils import save_clusters, load_clusters

def main():
    # --- CONFIGURATION ---
    TOPIC = "BLOCKCHAIN" 
    SAVE_PATH = "models/narrative_state.pkl"
    
    # Initialize components
    ingestor = GDELTIngestor()
    processor = DataProcessor(theme_filter=TOPIC)
    clusterer = IncrementalClusterer(threshold=0.65)

    # --- PERSISTENCE: Load existing memory if available ---
    if load_clusters(clusterer, SAVE_PATH):
        print(f" Loaded existing clusters from {SAVE_PATH}")
    else:
        print("🆕 Starting with a fresh narrative memory.")

    print(f" GDELT Narrative Streamer Active")
    print(f" Tracking Topic: {TOPIC}")
    print("-" * 50)

    while True:
        try:
            # 1. Fetch the latest 15-minute file
            data = ingestor.fetch_data()
            
            if data is not None:
                # 2. Filter and Vectorize
                df, vecs = processor.process(data)
                
                if vecs is not None:
                    # 3. Update clusters incrementally
                    total_clusters = clusterer.update(vecs, df)
                    
                    # 4. Save the state for the Dashboard (Streamlit)
                    save_clusters(clusterer, SAVE_PATH)
                    
                    print(f" [{time.strftime('%H:%M:%S')}] Processed {len(vecs)} articles.")
                    print(f" Total Unique Narratives in Memory: {total_clusters}")
                    
                    # 5. Display the top trending story
                    if clusterer.cluster_meta:
                        top_id = max(clusterer.cluster_meta, key=lambda k: clusterer.cluster_meta[k]['count'])
                        print(f" Current Top Trend: {clusterer.cluster_meta[top_id]['label']}")
                else:
                    print(f" [{time.strftime('%H:%M:%S')}] No '{TOPIC}' news in this window.")
            
            print("-" * 50)
            print(" Waiting 15 minutes for next GDELT heartbeat...")
            # GDELT updates every 15 mins; sleep for 900 seconds
            time.sleep(900)
            
        except KeyboardInterrupt:
            print("\n Shutting down stream gracefully...")
            save_clusters(clusterer, SAVE_PATH)
            sys.exit()
        except Exception as e:
            print(f" Unexpected Error: {e}")
            # Wait a minute before retrying to avoid spamming the CPU on errors
            time.sleep(60) 

if __name__ == "__main__":
    main()