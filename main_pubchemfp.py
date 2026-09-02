"""Entry point for training/evaluating the PubChemFP flat-vector ablation model."""

import logging
import sys
import argparse
from src.runtime import configure_thread_limits, configure_torch_runtime

configure_thread_limits()

import torch
configure_torch_runtime(torch)

from src.config import Config
from src.utils import set_seed, count_parameters
from src.pubchem_fp import PubChemFPLookup
from src.dataset import get_pubchemfp_dataloaders
from src.model_fp import APIExcipientFPModel
from src.loss import AsymmetricLoss
from src.train import train_model
from src.evaluate import (
    tune_threshold, run_final_evaluation, print_run_summary, save_run_metrics,
    calculate_metrics_at_threshold, evaluate_model_full, auc, precision_recall_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate the PubChemFP ablation model.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints/pubchemfp")
    parser.add_argument("--metrics-dir", default="metrics/pubchemfp")
    parser.add_argument("--descriptor-norm-stats-path", default=None)
    parser.add_argument("--api-pubchemfp-path", default=None)
    parser.add_argument("--excipient-pubchemfp-path", default=None)
    args = parser.parse_args()

    config = Config()
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.train_csv is not None:
        config.train_csv = args.train_csv
    if args.val_csv is not None:
        config.val_csv = args.val_csv
    if args.test_csv is not None:
        config.test_csv = args.test_csv
    config.checkpoint_dir = args.checkpoint_dir
    config.metrics_dir = args.metrics_dir
    if args.descriptor_norm_stats_path is not None:
        config.descriptor_norm_stats_path = args.descriptor_norm_stats_path
    if args.api_pubchemfp_path is not None:
        config.api_pubchemfp_path = args.api_pubchemfp_path
    if args.excipient_pubchemfp_path is not None:
        config.excipient_pubchemfp_path = args.excipient_pubchemfp_path

    config.positive_prior = config.compute_positive_prior()

    set_seed(config.seed)
    device = config.get_device()

    logging.info("=== API-Excipient Compatibility Prediction (PubChemFP ablation) ===")
    logging.info(f"Using device: {device}")

    fp_lookup = PubChemFPLookup(config.api_pubchemfp_path, config.excipient_pubchemfp_path)

    logging.info("Initializing datasets...")
    train_loader, val_loader, test_loader = get_pubchemfp_dataloaders(config, fp_lookup)
    logging.info(f"Batches per epoch: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")

    model = APIExcipientFPModel(config).to(device)
    trainable, total = count_parameters(model)
    logging.info(f"Model initialized. Trainable params: {trainable:,} / Total params: {total:,}")

    criterion = AsymmetricLoss(
        gamma_neg=config.asl_gamma_neg,
        gamma_pos=config.asl_gamma_pos,
        clip=config.asl_clip,
    )

    best_model = train_model(config, model, train_loader, val_loader, criterion)

    best_threshold = tune_threshold(best_model, val_loader, criterion, device, step=config.threshold_step)

    _, val_loss, val_true, val_prob = evaluate_model_full(best_model, val_loader, criterion, device)
    val_metrics = calculate_metrics_at_threshold(val_true, val_prob, best_threshold)
    precision, recall, _ = precision_recall_curve(val_true, val_prob)
    val_metrics["PR-AUC"] = auc(recall, precision)
    val_metrics["Loss"] = val_loss

    test_metrics = run_final_evaluation(best_model, test_loader, criterion, device, best_threshold)

    best_epoch = getattr(best_model, "best_epoch", None)
    print_run_summary(val_metrics, test_metrics, best_threshold, best_epoch=best_epoch)
    save_run_metrics(val_metrics, test_metrics, best_threshold, config.metrics_dir, best_epoch=best_epoch)


if __name__ == "__main__":
    main()
