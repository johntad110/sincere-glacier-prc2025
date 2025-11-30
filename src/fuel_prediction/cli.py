"""
Command-line interface for fuel prediction.
"""
import click
import os
from pathlib import Path

from .config import load_config
from .utils import setup_logger


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Fuel Prediction CLI - PRC 2025 Data Challenge"""
    pass


@cli.group()
def train():
    """Train models"""
    pass


@cli.group()
def predict():
    """Run inference"""
    pass


@cli.group()
def oof():
    """Generate out-of-fold predictions"""
    pass


@train.command('gbm')
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to config file')
@click.option('--data-dir', '-d', default='./prc-2025-datasets', help='Data directory')
@click.option('--output-dir', '-o', default='./models/gbm', help='Output directory')
def train_gbm(config, data_dir, output_dir):
    """Train LightGBM model"""
    click.echo("Training GBM model...")
    click.echo(f"Data directory: {data_dir}")
    click.echo(f"Output directory: {output_dir}")
    
    # Import here to avoid circular dependencies
    from pipelines import train_gbm_pipeline
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Run training (this will call the refactored pipeline script)
    click.echo("Running GBM training pipeline...")
    os.system(f"python pipelines/01_train_gbm.py")


@train.command('lstm')
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to config file')
@click.option('--data-dir', '-d', default='./prc-2025-datasets', help='Data directory')
@click.option('--output-dir', '-o', default='./models/lstm', help='Output directory')
def train_lstm(config, data_dir, output_dir):
    """Train LSTM model"""
    click.echo("Training LSTM model...")
    click.echo(f"Data directory: {data_dir}")
    click.echo(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    click.echo("Running LSTM training pipeline...")
    os.system(f"python pipelines/02_train_lstm.py")


@train.command('stacking')
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to config file')
def train_stacking(config):
    """Train stacking meta-model"""
    click.echo("Training stacking model...")
    
    click.echo("Running stacking training pipeline...")
    os.system(f"python pipelines/06_stacking_train.py")


@predict.command('gbm')
@click.option('--dataset', '-ds', type=click.Choice(['rank', 'final']), required=True, help='Dataset to predict on')
@click.option('--data-dir', '-d', default='./prc-2025-datasets', help='Data directory')
def predict_gbm(dataset, data_dir):
    """Run GBM inference"""
    click.echo(f"Running GBM inference on {dataset} dataset...")
    
    os.system(f"python pipelines/03_gbm_inference.py")


@predict.command('lstm')
@click.option('--dataset', '-ds', type=click.Choice(['rank', 'final']), required=True, help='Dataset to predict on')
def predict_lstm(dataset):
    """Run LSTM inference"""
    click.echo(f"Running LSTM inference on {dataset} dataset...")
    
    os.system(f"python pipelines/04_lstm_inference.py")


@predict.command('stacking')
@click.option('--dataset', '-ds', type=click.Choice(['rank', 'final']), required=True, help='Dataset to predict on')
def predict_stacking(dataset):
    """Run stacking inference"""
    click.echo(f"Running stacking inference on {dataset} dataset...")
    
    os.system(f"python pipelines/07_stacking_inference.py")


@oof.command('gbm')
def oof_gbm():
    """Generate GBM out-of-fold predictions"""
    click.echo("Generating GBM OOF predictions...")
    click.echo("Note: GBM OOF is generated during training")
    click.echo("Check models/gbm/oof_predictions.parquet")


@oof.command('lstm')
def oof_lstm():
    """Generate LSTM out-of-fold predictions"""
    click.echo("Generating LSTM OOF predictions...")
    
    os.system(f"python pipelines/05_lstm_oof.py")


@cli.command()
def pipeline():
    """Run full training and inference pipeline"""
    click.echo("Running complete pipeline...")
    click.echo("\n=== Phase 1: Model Training ===")
    
    click.echo("\n[1/5] Training GBM...")
    os.system("python pipelines/01_train_gbm.py")
    
    click.echo("\n[2/5] Training LSTM...")
    os.system("python pipelines/02_train_lstm.py")
    
    click.echo("\n[3/5] Generating LSTM OOF...")
    os.system("python pipelines/05_lstm_oof.py")
    
    click.echo("\n[4/5] Training Stacking...")
    os.system("python pipelines/06_stacking_train.py")
    
    click.echo("\n[5/5] Running inference...")
    os.system("python pipelines/07_stacking_inference.py")
    
    click.echo("\n✅ Pipeline complete! Check submissions/ folder for results.")


if __name__ == '__main__':
    cli()
