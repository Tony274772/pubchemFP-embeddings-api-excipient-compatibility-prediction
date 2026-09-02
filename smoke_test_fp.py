"""Smoke test for PubChemFP model: minimal sanity check before full training."""

import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def smoke_test_fp():
    """Run basic smoke test: load config, lookup, dataloader, and forward pass."""
    
    try:
        logger.info("1. Loading Config...")
        from src.config import Config
        config = Config()
        logger.info("   ✓ Config loaded")
        
        logger.info("2. Loading PubChemFPLookup...")
        from src.pubchem_fp import PubChemFPLookup
        fp_lookup = PubChemFPLookup(config.api_pubchemfp_path, config.excipient_pubchemfp_path)
        logger.info("   ✓ PubChemFPLookup loaded")
        
        logger.info("3. Building one batch via get_pubchemfp_dataloaders...")
        from src.dataset import get_pubchemfp_dataloaders
        train_loader, val_loader, test_loader = get_pubchemfp_dataloaders(config, fp_lookup)
        logger.info(f"   ✓ Dataloaders created (train batches: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)})")
        
        logger.info("4. Getting first batch from train loader...")
        batch = next(iter(train_loader))
        logger.info(f"   ✓ Batch retrieved: api_fp shape {batch['api_fp'].shape}, labels shape {batch['labels'].shape}")
        
        logger.info("5. Building APIExcipientFPModel...")
        import torch
        from src.model_fp import APIExcipientFPModel
        model = APIExcipientFPModel(config)
        device = config.get_device()
        model = model.to(device)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        logger.info("   ✓ Model built and moved to device")
        
        logger.info("6. Running forward pass...")
        logits = model(batch)
        logger.info(f"   ✓ Forward pass successful, logits shape: {logits.shape}")
        
        logger.info("7. Validating output tensor...")
        assert logits.shape == (batch["labels"].shape[0],), f"Expected shape ({batch['labels'].shape[0]},), got {logits.shape}"
        assert not torch.isnan(logits).any(), "Output contains NaN values!"
        assert not torch.isinf(logits).any(), "Output contains Inf values!"
        logger.info("   ✓ Output tensor valid (no NaNs, correct shape)")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ SMOKE TEST PASSED")
        logger.info("=" * 60)
        return True
    
    except Exception as e:
        logger.error(f"\n❌ SMOKE TEST FAILED: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = smoke_test_fp()
    sys.exit(0 if success else 1)
