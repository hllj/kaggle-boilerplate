"""
Script to generate submission files for Kaggle competitions.
"""
import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Union

import pandas as pd

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config, get_config
from src.models.inference import generate_predictions, save_submission
from src.utils import setup_logging


def make_submission(cfg: Config, logger: logging.Logger) -> str:
    """
    Generate submission file.
    
    Args:
        cfg: Configuration
        logger: Logger
        
    Returns:
        Path to submission file
    """
    # Generate predictions
    logger.info("Generating predictions...")
    submission_df = generate_predictions(cfg, logger)
    
    # Save submission
    logger.info("Saving submission...")
    submission_filepath = save_submission(submission_df, cfg)
    
    return submission_filepath


def upload_to_kaggle(submission_filepath: str, 
                    competition_name: str, 
                    message: str, 
                    logger: logging.Logger) -> None:
    """
    Upload submission to Kaggle.
    
    Args:
        submission_filepath: Path to submission file
        competition_name: Name of Kaggle competition
        message: Submission message
        logger: Logger
    """
    try:
        import kaggle
    except ImportError:
        logger.error("Kaggle API not found. Please install it with 'pip install kaggle'.")
        return
    
    # Upload submission
    logger.info(f"Uploading submission to Kaggle competition '{competition_name}'...")
    try:
        # Check if Kaggle API credentials are set up
        kaggle_dir = os.path.expanduser('~/.kaggle')
        kaggle_api_path = os.path.join(kaggle_dir, 'kaggle.json')
        
        if not os.path.exists(kaggle_api_path):
            logger.error(
                "Kaggle API credentials not found. Please run 'kaggle competitions download -c "
                f"{competition_name}' to authenticate and download the competition data."
            )
            return
        
        # Make API call to upload submission
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        api.competition_submit(
            file_name=submission_filepath,
            message=message,
            competition=competition_name
        )
        
        logger.info("Submission uploaded successfully!")
        
    except Exception as e:
        logger.error(f"Error uploading submission to Kaggle: {str(e)}")


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate submission for Kaggle competition")
    parser.add_argument("--config", type=str, default="default", help="Name of config file")
    parser.add_argument("--upload", action="store_true", help="Upload submission to Kaggle")
    parser.add_argument("--competition", type=str, help="Kaggle competition name")
    parser.add_argument("--message", type=str, default="Submission from CLI", help="Submission message")
    args = parser.parse_args()
    
    # Load configuration
    cfg = get_config(args.config)
    
    # Set up logging
    logger = setup_logging(cfg)
    
    # Generate submission
    submission_filepath = make_submission(cfg, logger)
    logger.info(f"Submission saved to {submission_filepath}")
    
    # Upload to Kaggle if requested
    if args.upload:
        if not args.competition:
            logger.error("Competition name must be specified with --competition for upload")
            return
        
        upload_to_kaggle(
            submission_filepath=submission_filepath,
            competition_name=args.competition,
            message=args.message,
            logger=logger
        )


if __name__ == "__main__":
    main() 