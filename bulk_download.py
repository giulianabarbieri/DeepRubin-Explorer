import os
import time
from src.ingestion import DataIngestor

def main():
    """
    Main execution script to build the initial training dataset.
    Downloads multiple astronomical classes for comparative analysis.
    """
    ingestor = DataIngestor(output_dir="data")
    
    # Define classes of interest for our classification model
    # SNIa: Supernovae Type Ia (Standard Candles)
    # SNII: Supernovae Type II (Core Collapse)
    # AGN: Active Galactic Nuclei (Stochastic variability)
    classes_to_download = ["SNIa", "SNII", "AGN"]
    samples_per_class = 50 

    print("--- Starting Bulk Download for DeepRubin-Explorer ---")

    for class_name in classes_to_download:
        print(f"\n[!] Processing class: {class_name}")
        
        # 1. Fetch metadata for the targets
        targets = ingestor.fetch_sample_targets(class_name, count=samples_per_class)
        
        # 2. Iterate and download light curves
        count = 0
        for oid in targets['oid']:
            success = ingestor.download_and_save(oid)
            if success:
                count += 1
            
            # Brief sleep to be respectful to the ALeRCE API rate limits
            time.sleep(0.5) 
            
        print(f"[✔] Finished {class_name}. Successfully downloaded: {count}/{samples_per_class}")

    print("\n--- Dataset acquisition complete. Check the 'data/' folder. ---")

if __name__ == "__main__":
    main()