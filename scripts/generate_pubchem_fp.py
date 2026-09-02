"""Generate PubChem CACTVS fingerprints for all unique CIDs in the dataset."""

import os
import sys
import time
import logging
from pathlib import Path

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def load_unique_cids(data_dir="data"):
    """Load unique API and excipient CIDs from train/val/test CSVs."""
    api_cids = set()
    exc_cids = set()
    
    for csv_file in ["train.csv", "val.csv", "test.csv"]:
        csv_path = os.path.join(data_dir, csv_file)
        if not os.path.exists(csv_path):
            logging.warning(f"Skipping {csv_path} (not found)")
            continue
        
        df = pd.read_csv(csv_path)
        api_cids.update(df["API_CID"].unique())
        exc_cids.update(df["Excipient_CID"].unique())
        logging.info(f"Loaded CIDs from {csv_file}: {len(df)} rows")
    
    return sorted(api_cids), sorted(exc_cids)


def fetch_pubchem_fp(cid, max_retries=3):
    """Fetch PubChem CACTVS fingerprint for a single CID.
    
    Args:
        cid: PubChem Compound ID (int)
        max_retries: number of retry attempts with exponential backoff
        
    Returns:
        (success: bool, fp_string: str or None, error_msg: str or None)
    """
    import pubchempy as pcp
    
    backoff_times = [0.5, 1.0, 2.0]  # exponential backoff: 0.5s, 1s, 2s
    
    for attempt in range(max_retries):
        try:
            compound = pcp.Compound.from_cid(int(cid))
            fp_string = compound.cactvs_fingerprint
            
            # Validate immediately
            if not isinstance(fp_string, str):
                return False, None, f"CID {cid}: fingerprint is not a string, got {type(fp_string)}"
            
            if len(fp_string) != 881:
                return False, None, f"CID {cid}: fingerprint length {len(fp_string)}, expected 881"
            
            if not set(fp_string) <= {"0", "1"}:
                return False, None, f"CID {cid}: fingerprint contains invalid characters (expected 0/1)"
            
            return True, fp_string, None
        
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                wait_time = backoff_times[attempt]
                logging.debug(f"CID {cid}: attempt {attempt + 1} failed, retrying in {wait_time}s... ({error_msg})")
                time.sleep(wait_time)
            else:
                return False, None, f"CID {cid}: all {max_retries} attempts failed: {error_msg}"
    
    return False, None, f"CID {cid}: unknown error after {max_retries} attempts"


def generate_pubchem_fp(data_dir="data", api_output="data/api_pubchemfp.csv", exc_output="data/excipient_pubchemfp.csv"):
    """Generate PubChem fingerprints for all unique CIDs in the dataset."""
    
    logging.info("Loading unique CIDs from train/val/test CSVs...")
    api_cids, exc_cids = load_unique_cids(data_dir)
    logging.info(f"Total unique API CIDs: {len(api_cids)}")
    logging.info(f"Total unique excipient CIDs: {len(exc_cids)}")
    
    # Collect all unique CIDs and their type (for tracking)
    all_cids_to_fetch = {}
    for cid in api_cids:
        if cid not in all_cids_to_fetch:
            all_cids_to_fetch[cid] = []
        all_cids_to_fetch[cid].append("API")
    for cid in exc_cids:
        if cid not in all_cids_to_fetch:
            all_cids_to_fetch[cid] = []
        all_cids_to_fetch[cid].append("Excipient")
    
    total_unique_cids = len(all_cids_to_fetch)
    logging.info(f"Total unique CIDs across all types: {total_unique_cids}")
    
    # Fetch fingerprints
    logging.info("Fetching PubChem CACTVS fingerprints (rate limited to 5 req/sec)...")
    
    cid_to_fp = {}
    failed_cids = []
    
    for idx, cid in enumerate(sorted(all_cids_to_fetch.keys())):
        if idx > 0:
            # Rate limiting: 5 requests/second = 1 request per 0.2s, use 0.25s to be safe
            time.sleep(0.25)
        
        if (idx + 1) % 100 == 0:
            logging.info(f"Fetching {idx + 1}/{total_unique_cids}...")
        
        success, fp_string, error_msg = fetch_pubchem_fp(cid)
        
        if success:
            cid_to_fp[cid] = fp_string
        else:
            logging.warning(error_msg)
            failed_cids.append(cid)
    
    logging.info(f"Fetching complete. Succeeded: {len(cid_to_fp)}, Failed: {len(failed_cids)}")
    if failed_cids:
        logging.warning(f"Failed CIDs: {failed_cids}")
    
    # Convert to dataframes
    logging.info("Converting fingerprints to dataframes...")
    
    api_rows = []
    for cid in sorted(api_cids):
        if cid not in cid_to_fp:
            logging.error(f"Missing fingerprint for API CID {cid}")
            return False
        
        fp_string = cid_to_fp[cid]
        row = {"API_CID": cid}
        for i, bit in enumerate(fp_string):
            row[f"fp_{i:04d}"] = int(bit)
        api_rows.append(row)
    
    exc_rows = []
    for cid in sorted(exc_cids):
        if cid not in cid_to_fp:
            logging.error(f"Missing fingerprint for excipient CID {cid}")
            return False
        
        fp_string = cid_to_fp[cid]
        row = {"Excipient_CID": cid}
        for i, bit in enumerate(fp_string):
            row[f"fp_{i:04d}"] = int(bit)
        exc_rows.append(row)
    
    api_df = pd.DataFrame(api_rows)
    exc_df = pd.DataFrame(exc_rows)
    
    # Verify and save
    logging.info(f"API DataFrame shape: {api_df.shape}")
    logging.info(f"Excipient DataFrame shape: {exc_df.shape}")
    
    # Check column counts
    expected_cols = 1 + 881  # CID column + 881 fingerprint columns
    if len(api_df.columns) != expected_cols:
        logging.error(f"API DataFrame has {len(api_df.columns)} columns, expected {expected_cols}")
        return False
    if len(exc_df.columns) != expected_cols:
        logging.error(f"Excipient DataFrame has {len(exc_df.columns)} columns, expected {expected_cols}")
        return False
    
    # Save to CSV
    os.makedirs(os.path.dirname(api_output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(exc_output) or ".", exist_ok=True)
    
    api_df.to_csv(api_output, index=False)
    exc_df.to_csv(exc_output, index=False)
    
    logging.info(f"Saved API fingerprints to {api_output}")
    logging.info(f"Saved excipient fingerprints to {exc_output}")
    
    # Final summary
    logging.info("=" * 60)
    logging.info("PubChem fingerprint generation complete")
    logging.info(f"Total unique API CIDs: {len(api_cids)}")
    logging.info(f"Total unique excipient CIDs: {len(exc_cids)}")
    logging.info(f"Successfully fetched: {len(cid_to_fp)}")
    logging.info(f"Failed fetches: {len(failed_cids)}")
    if failed_cids:
        logging.info(f"Failed CID list: {failed_cids}")
    logging.info("=" * 60)
    
    return True


if __name__ == "__main__":
    success = generate_pubchem_fp()
    sys.exit(0 if success else 1)
