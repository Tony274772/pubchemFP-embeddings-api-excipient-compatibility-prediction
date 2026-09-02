"""PubChem fingerprint lookup and loading."""

import numpy as np
import pandas as pd


PUBCHEM_FP_DIM = 881


def _cid_key(cid):
    return str(cid)


class PubChemFPLookup:
    """Loads and provides access to precomputed PubChem CACTVS fingerprints.
    
    Fingerprints are stored as CSVs keyed by CID, with 881 binary columns.
    Raises ValueError on missing CID lookups (fail loudly, never zero-fill).
    """
    
    def __init__(self, api_csv_path: str, exc_csv_path: str):
        """Load fingerprint CSVs into memory.
        
        Args:
            api_csv_path: path to data/api_pubchemfp.csv
            exc_csv_path: path to data/excipient_pubchemfp.csv
        """
        self.api_lookup = self._load_fp_lookup(api_csv_path, "API_CID")
        self.exc_lookup = self._load_fp_lookup(exc_csv_path, "Excipient_CID")
    
    def _load_fp_lookup(self, csv_path, cid_col):
        """Load fingerprint CSV into dict keyed by str(cid) -> np.ndarray shape (881,), dtype float32."""
        df = pd.read_csv(csv_path)
        
        if cid_col not in df.columns:
            raise ValueError(f"{csv_path} is missing {cid_col} column")
        
        # Extract fingerprint columns (everything except the CID column)
        fp_cols = [col for col in df.columns if col != cid_col]
        if len(fp_cols) != PUBCHEM_FP_DIM:
            raise ValueError(
                f"{csv_path} has {len(fp_cols)} fingerprint columns, "
                f"expected {PUBCHEM_FP_DIM}"
            )
        
        lookup = {}
        for _, row in df.iterrows():
            cid = _cid_key(row[cid_col])
            fp_array = row[fp_cols].to_numpy(dtype=np.float32)
            lookup[cid] = fp_array
        
        return lookup
    
    def get_api(self, api_cid) -> np.ndarray:
        """Get API fingerprint by CID.
        
        Args:
            api_cid: API compound ID (int or str)
            
        Returns:
            np.ndarray of shape (881,) and dtype float32
            
        Raises:
            ValueError if CID not found
        """
        key = _cid_key(api_cid)
        if key not in self.api_lookup:
            raise ValueError(f"Missing API PubChemFP row for CID {api_cid}")
        return self.api_lookup[key]
    
    def get_exc(self, exc_cid) -> np.ndarray:
        """Get excipient fingerprint by CID.
        
        Args:
            exc_cid: excipient compound ID (int or str)
            
        Returns:
            np.ndarray of shape (881,) and dtype float32
            
        Raises:
            ValueError if CID not found
        """
        key = _cid_key(exc_cid)
        if key not in self.exc_lookup:
            raise ValueError(f"Missing excipient PubChemFP row for CID {exc_cid}")
        return self.exc_lookup[key]
