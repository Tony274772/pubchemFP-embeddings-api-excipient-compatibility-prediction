"""Generate PubChem CACTVS fingerprints for all unique CIDs in the dataset.

Fixes over the previous version:
  - Sets a hard socket timeout so a single stalled PubChem request can't hang
    the whole script forever (this was the most likely cause of it appearing
    to "stop").
  - Logs progress for EVERY CID (index, CID, elapsed time, success/failure),
    not just every 100 — so you can see it's alive in real time.
  - Writes results incrementally to a checkpoint CSV after every fetch, and
    resumes from it on restart, so interrupting the script (Ctrl+C, crash,
    network drop) doesn't lose progress already fetched.
"""

import os
import sys
import time
import socket
import logging
from pathlib import Path

import pandas as pd
import numpy as np

# --- Hard timeout for ALL network calls (pubchempy uses urllib under the ---
# --- hood, which has no timeout by default -> this is what makes a single ---
# --- bad request hang forever with zero output). ---------------------------
socket.setdefaulttimeout(15)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Force line-by-line flushing so log lines show up immediately in your
# terminal instead of being buffered.
for handler in logging.getLogger().handlers:
    handler.flush = lambda: sys.stdout.flush()

CHECKPOINT_PATH = "data/_pubchem_fp_checkpoint.csv"


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


def load_checkpoint(checkpoint_path=CHECKPOINT_PATH):
    """Load already-fetched CID -> fingerprint pairs from a previous run, if any."""
    if not os.path.exists(checkpoint_path):
        return {}
    df = pd.read_csv(checkpoint_path, dtype={"cid": int, "fp_string": str})
    cid_to_fp = dict(zip(df["cid"], df["fp_string"]))
    logging.info(f"Resuming from checkpoint: {len(cid_to_fp)} CIDs already fetched.")
    return cid_to_fp


def append_checkpoint(cid, fp_string, checkpoint_path=CHECKPOINT_PATH):
    """Append one successful fetch to the checkpoint file immediately."""
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    write_header = not os.path.exists(checkpoint_path)
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write("cid,fp_string\n")
        f.write(f"{cid},{fp_string}\n")


def fetch_pubchem_fp(cid, max_retries=3):
    """Fetch PubChem CACTVS fingerprint for a single CID.

    Returns:
        (success: bool, fp_string: str or None, error_msg: str or None)
    """
    import pubchempy as pcp

    backoff_times = [0.5, 1.0, 2.0]

    for attempt in range(max_retries):
        try:
            compound = pcp.Compound.from_cid(int(cid))
            fp_string = compound.cactvs_fingerprint

            if not isinstance(fp_string, str):
                return False, None, f"fingerprint is not a string, got {type(fp_string)}"
            if len(fp_string) != 881:
                return False, None, f"fingerprint length {len(fp_string)}, expected 881"
            if not set(fp_string) <= {"0", "1"}:
                return False, None, "fingerprint contains invalid characters (expected 0/1)"

            return True, fp_string, None

        except socket.timeout:
            error_msg = f"timed out after {socket.getdefaulttimeout()}s"
        except Exception as e:
            error_msg = str(e)

        if attempt < max_retries - 1:
            wait_time = backoff_times[attempt]
            logging.warning(
                f"  CID {cid}: attempt {attempt + 1}/{max_retries} failed ({error_msg}), "
                f"retrying in {wait_time}s..."
            )
            time.sleep(wait_time)
        else:
            return False, None, f"all {max_retries} attempts failed: {error_msg}"

    return False, None, "unknown error"


def generate_pubchem_fp(data_dir="data", api_output="data/api_pubchemfp.csv", exc_output="data/excipient_pubchemfp.csv"):
    """Generate PubChem fingerprints for all unique CIDs in the dataset."""

    logging.info("Loading unique CIDs from train/val/test CSVs...")
    api_cids, exc_cids = load_unique_cids(data_dir)
    logging.info(f"Total unique API CIDs: {len(api_cids)}")
    logging.info(f"Total unique excipient CIDs: {len(exc_cids)}")

    all_cids = sorted(set(api_cids) | set(exc_cids))
    total_unique_cids = len(all_cids)
    logging.info(f"Total unique CIDs across all types: {total_unique_cids}")

    # Resume from a previous partial run instead of re-fetching everything.
    cid_to_fp = load_checkpoint()
    remaining = [cid for cid in all_cids if cid not in cid_to_fp]
    logging.info(f"Already have {len(cid_to_fp)} cached; {len(remaining)} left to fetch.")

    failed_cids = []
    start_time = time.time()

    logging.info("Fetching PubChem CACTVS fingerprints (rate limited to ~4 req/sec)...")

    for i, cid in enumerate(remaining, start=1):
        t0 = time.time()
        success, fp_string, error_msg = fetch_pubchem_fp(cid)
        elapsed = time.time() - t0
        overall_done = len(cid_to_fp) + len(failed_cids) + 1

        if success:
            cid_to_fp[cid] = fp_string
            append_checkpoint(cid, fp_string)
            logging.info(
                f"[{overall_done}/{total_unique_cids}] CID {cid}: OK "
                f"({elapsed:.2f}s)"
            )
        else:
            failed_cids.append(cid)
            logging.error(
                f"[{overall_done}/{total_unique_cids}] CID {cid}: FAILED - {error_msg}"
            )

        # Rate limit: PubChem asks for <=5 req/sec; keep some margin.
        time.sleep(0.25)

        # Periodic throughput summary every 25 CIDs so you can see it's not stalled.
        if i % 25 == 0:
            rate = i / (time.time() - start_time)
            logging.info(f"  -> throughput: {rate:.2f} CIDs/sec, "
                         f"{len(remaining) - i} remaining (~{(len(remaining) - i) / max(rate, 0.01):.0f}s left)")

    logging.info(f"Fetching complete. Succeeded: {len(cid_to_fp)}, Failed: {len(failed_cids)}")
    if failed_cids:
        logging.warning(f"Failed CIDs (not cached, will retry on next run): {failed_cids}")

    # Only build the final wide CSVs once every needed CID is present.
    missing_api = [cid for cid in api_cids if cid not in cid_to_fp]
    missing_exc = [cid for cid in exc_cids if cid not in cid_to_fp]
    if missing_api or missing_exc:
        logging.error(
            f"Cannot write final output yet: {len(missing_api)} API CIDs and "
            f"{len(missing_exc)} excipient CIDs still missing fingerprints. "
            f"Re-run this script to resume (already-fetched CIDs are cached in "
            f"{CHECKPOINT_PATH} and will not be re-fetched)."
        )
        return False

    logging.info("All required CIDs fetched. Building output CSVs...")

    api_rows = []
    for cid in sorted(api_cids):
        fp_string = cid_to_fp[cid]
        row = {"API_CID": cid}
        for i, bit in enumerate(fp_string):
            row[f"fp_{i:04d}"] = int(bit)
        api_rows.append(row)

    exc_rows = []
    for cid in sorted(exc_cids):
        fp_string = cid_to_fp[cid]
        row = {"Excipient_CID": cid}
        for i, bit in enumerate(fp_string):
            row[f"fp_{i:04d}"] = int(bit)
        exc_rows.append(row)

    api_df = pd.DataFrame(api_rows)
    exc_df = pd.DataFrame(exc_rows)

    logging.info(f"API DataFrame shape: {api_df.shape}")
    logging.info(f"Excipient DataFrame shape: {exc_df.shape}")

    expected_cols = 1 + 881
    if len(api_df.columns) != expected_cols:
        logging.error(f"API DataFrame has {len(api_df.columns)} columns, expected {expected_cols}")
        return False
    if len(exc_df.columns) != expected_cols:
        logging.error(f"Excipient DataFrame has {len(exc_df.columns)} columns, expected {expected_cols}")
        return False

    os.makedirs(os.path.dirname(api_output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(exc_output) or ".", exist_ok=True)

    api_df.to_csv(api_output, index=False)
    exc_df.to_csv(exc_output, index=False)

    logging.info(f"Saved API fingerprints to {api_output}")
    logging.info(f"Saved excipient fingerprints to {exc_output}")

    logging.info("=" * 60)
    logging.info("PubChem fingerprint generation complete")
    logging.info(f"Total unique API CIDs: {len(api_cids)}")
    logging.info(f"Total unique excipient CIDs: {len(exc_cids)}")
    logging.info(f"Successfully fetched (this run + cache): {len(cid_to_fp)}")
    logging.info(f"Failed fetches this run: {len(failed_cids)}")
    logging.info("=" * 60)

    return True


if __name__ == "__main__":
    success = generate_pubchem_fp()
    sys.exit(0 if success else 1)