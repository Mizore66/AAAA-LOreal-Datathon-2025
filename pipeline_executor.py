#!/usr/bin/env python3
"""
External Agent Pipeline Executor for L'Oréal Datathon 2025

This script can be delegated to external agents for automated pipeline execution.
It runs the complete data processing pipeline and automatically commits generated files.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INTERIM_DIR = DATA_DIR / "interim"


class PipelineExecutor:
    """External agent pipeline executor with git automation."""
    
    def __init__(self, auto_commit: bool = True, skip_existing: bool = True):
        self.auto_commit = auto_commit
        self.skip_existing = skip_existing
        self.execution_start = datetime.now()
        self.generated_files = []
        
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met before running pipeline."""
        logger.info("Checking prerequisites...")
        
        # Check if we're in a git repository
        if not (ROOT / ".git").exists():
            logger.error("Not in a git repository")
            return False
            
        # Check if source files exist
        required_files = [
            SRC_DIR / "ingest_provided_data.py",
            SRC_DIR / "data_processing.py",
            SRC_DIR / "modeling.py"
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                logger.error(f"Required file missing: {file_path}")
                return False
                
        # Check if Python environment has required packages
        try:
            import pandas
            import numpy
            import pyarrow
            logger.info("Core dependencies available")
        except ImportError as e:
            logger.error(f"Missing required Python package: {e}")
            return False
            
        # Ensure data directories exist
        for dir_path in [DATA_DIR, PROCESSED_DIR, INTERIM_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        logger.info("Prerequisites check passed")
        return True
        
    def run_ingestion(self) -> bool:
        """Run data ingestion pipeline."""
        logger.info("Running data ingestion...")
        
        try:
            result = subprocess.run(
                [sys.executable, str(SRC_DIR / "ingest_provided_data.py")],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info("Data ingestion completed successfully")
                logger.debug(f"Ingestion output: {result.stdout}")
                return True
            else:
                logger.error(f"Data ingestion failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Data ingestion timed out")
            return False
        except Exception as e:
            logger.error(f"Data ingestion error: {e}")
            return False
            
    def run_data_processing(self) -> bool:
        """Run data processing and feature engineering."""
        logger.info("Running data processing and feature engineering...")
        
        try:
            result = subprocess.run(
                [sys.executable, str(SRC_DIR / "data_processing.py")],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info("Data processing completed successfully")
                logger.debug(f"Processing output: {result.stdout}")
                return True
            else:
                logger.error(f"Data processing failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Data processing timed out")
            return False
        except Exception as e:
            logger.error(f"Data processing error: {e}")
            return False
            
    def run_modeling(self) -> bool:
        """Run modeling pipeline."""
        logger.info("Running modeling pipeline...")
        
        # Check if we have enough dependencies for modeling
        try:
            import prophet
            import sklearn
            modeling_available = True
        except ImportError:
            logger.warning("Advanced modeling dependencies not available, skipping modeling")
            modeling_available = False
            
        if not modeling_available:
            return True  # Skip modeling but don't fail the pipeline
            
        try:
            result = subprocess.run(
                [sys.executable, str(SRC_DIR / "modeling.py")],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info("Modeling completed successfully")
                logger.debug(f"Modeling output: {result.stdout}")
                return True
            else:
                logger.warning(f"Modeling had issues but continuing: {result.stderr}")
                return True  # Don't fail pipeline for modeling issues
                
        except subprocess.TimeoutExpired:
            logger.warning("Modeling timed out, continuing pipeline")
            return True
        except Exception as e:
            logger.warning(f"Modeling error, continuing pipeline: {e}")
            return True
            
    def collect_generated_files(self) -> list:
        """Collect all files generated by the pipeline."""
        generated_files = []
        
        # Check processed data directory
        if PROCESSED_DIR.exists():
            for file_path in PROCESSED_DIR.glob("*.parquet"):
                if file_path.stat().st_mtime > self.execution_start.timestamp():
                    generated_files.append(file_path)
                    
        # Check interim reports
        if INTERIM_DIR.exists():
            for file_path in INTERIM_DIR.glob("*.md"):
                if file_path.stat().st_mtime > self.execution_start.timestamp():
                    generated_files.append(file_path)
                    
        # Check models directory
        models_dir = ROOT / "models"
        if models_dir.exists():
            for file_path in models_dir.glob("*.pkl"):
                if file_path.stat().st_mtime > self.execution_start.timestamp():
                    generated_files.append(file_path)
                    
        self.generated_files = generated_files
        logger.info(f"Found {len(generated_files)} newly generated files")
        return generated_files
        
    def git_add_and_commit(self, commit_message: str = None) -> bool:
        """Add generated files to git and commit."""
        if not self.auto_commit:
            logger.info("Auto-commit disabled, skipping git operations")
            return True
            
        if not self.generated_files:
            logger.info("No new files to commit")
            return True
            
        logger.info("Adding generated files to git...")
        
        try:
            # Create a summary of results instead of committing large data files
            summary_data = self.create_results_summary()
            
            # Add pipeline files that should be committed
            committable_files = [
                "pipeline_executor.py",
                "test_pipeline.py"
            ]
            
            files_added = []
            for file_name in committable_files:
                file_path = ROOT / file_name
                if file_path.exists():
                    result = subprocess.run(
                        ["git", "add", str(file_path.relative_to(ROOT))],
                        cwd=ROOT,
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        files_added.append(file_name)
                    else:
                        logger.warning(f"Failed to add {file_name}: {result.stderr}")
            
            # Add the execution summary
            summary_path = ROOT / "execution_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            result = subprocess.run(
                ["git", "add", "execution_summary.json"],
                cwd=ROOT,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                files_added.append("execution_summary.json")
                    
            # Create commit message
            if not commit_message:
                file_count = len(self.generated_files)
                timestamp = self.execution_start.strftime("%Y-%m-%d %H:%M:%S")
                commit_message = f"Pipeline execution: {file_count} data files generated, pipeline updates ({timestamp})"
                
            # Only commit if we have files to commit
            if files_added:
                result = subprocess.run(
                    ["git", "commit", "-m", commit_message],
                    cwd=ROOT,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    logger.info(f"Successfully committed changes: {commit_message}")
                    logger.info(f"Files committed: {', '.join(files_added)}")
                    return True
                else:
                    logger.warning(f"Git commit had issues: {result.stderr}")
                    return False
            else:
                logger.info("No committable files found (data files are in .gitignore)")
                return True
                
        except Exception as e:
            logger.error(f"Git operations failed: {e}")
            return False
            
    def create_results_summary(self) -> dict:
        """Create a summary of generated results."""
        summary = {
            "execution_timestamp": self.execution_start.isoformat(),
            "pipeline_version": "1.0",
            "generated_files_count": len(self.generated_files),
            "generated_files_summary": {},
            "feature_timeframes": [],
            "reports_generated": []
        }
        
        # Categorize generated files
        for file_path in self.generated_files:
            file_name = file_path.name
            file_category = "other"
            
            if "features_" in file_name:
                file_category = "feature_aggregation"
                # Extract timeframe
                for timeframe in ['1h', '3h', '6h', '1d', '3d', '7d', '14d', '1m', '3m', '6m']:
                    if f"_{timeframe}.parquet" in file_name:
                        if timeframe not in summary["feature_timeframes"]:
                            summary["feature_timeframes"].append(timeframe)
                        break
            elif "trend_clusters" in file_name:
                file_category = "trend_analysis"
            elif "emerging_terms" in file_name:
                file_category = "emerging_trends"
            elif "anomalies" in file_name:
                file_category = "anomaly_detection"
            elif file_name.endswith(".md"):
                file_category = "reports"
                summary["reports_generated"].append(file_name)
                
            if file_category not in summary["generated_files_summary"]:
                summary["generated_files_summary"][file_category] = 0
            summary["generated_files_summary"][file_category] += 1
            
        return summary
            
    def create_execution_report(self) -> dict:
        """Create a comprehensive execution report."""
        execution_end = datetime.now()
        duration = execution_end - self.execution_start
        
        report = {
            "execution_id": self.execution_start.strftime("%Y%m%d_%H%M%S"),
            "start_time": self.execution_start.isoformat(),
            "end_time": execution_end.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "generated_files": [str(f.relative_to(ROOT)) for f in self.generated_files],
            "file_count": len(self.generated_files),
            "auto_commit_enabled": self.auto_commit,
            "skip_existing_enabled": self.skip_existing
        }
        
        # Save report
        report_path = INTERIM_DIR / f"pipeline_execution_{report['execution_id']}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Execution report saved: {report_path}")
        return report
        
    def execute_full_pipeline(self) -> bool:
        """Execute the complete pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING PIPELINE EXECUTION")
        logger.info("=" * 60)
        
        # Check prerequisites
        if not self.check_prerequisites():
            logger.error("Prerequisites check failed")
            return False
            
        success = True
        
        # Run ingestion
        if not self.run_ingestion():
            logger.error("Ingestion failed")
            success = False
            
        # Run data processing (even if ingestion had issues)
        if not self.run_data_processing():
            logger.error("Data processing failed")
            success = False
            
        # Run modeling (optional)
        self.run_modeling()  # Don't fail pipeline for modeling issues
        
        # Collect generated files
        self.collect_generated_files()
        
        # Git operations
        if self.auto_commit:
            self.git_add_and_commit()
            
        # Create execution report
        report = self.create_execution_report()
        
        logger.info("=" * 60)
        if success:
            logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        else:
            logger.warning("PIPELINE EXECUTION COMPLETED WITH ISSUES")
        logger.info(f"Generated {len(self.generated_files)} files")
        logger.info(f"Execution time: {report['duration_seconds']:.1f} seconds")
        logger.info("=" * 60)
        
        return success


def main():
    """Main entry point for pipeline executor."""
    parser = argparse.ArgumentParser(
        description="External Agent Pipeline Executor for L'Oréal Datathon 2025"
    )
    parser.add_argument(
        "--no-commit", 
        action="store_true", 
        help="Disable automatic git commit of generated files"
    )
    parser.add_argument(
        "--force-regenerate", 
        action="store_true", 
        help="Force regeneration of existing files"
    )
    parser.add_argument(
        "--commit-message", 
        type=str, 
        help="Custom commit message for generated files"
    )
    
    args = parser.parse_args()
    
    # Initialize executor
    executor = PipelineExecutor(
        auto_commit=not args.no_commit,
        skip_existing=not args.force_regenerate
    )
    
    # Execute pipeline
    success = executor.execute_full_pipeline()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()