"""
Setup script for fuel-prediction package.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="fuel-prediction",
    version="1.0.0",
    author="Team Sincere Glacier",
    description="Aircraft fuel burn prediction using hybrid stacking (GBM + LSTM)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/johntad110/sincere-glacier-prc2025",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={
        "fuel_prediction": ["config/*.yaml"],
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "lightgbm>=4.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "click>=8.1.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fuel-predict=fuel_prediction.cli:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="GNU GPLv3",
)
