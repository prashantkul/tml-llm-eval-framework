"""Setup script for LLM Comprehensive Evaluation Pipeline."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    with open(this_directory / "requirements.txt") as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#') and not line.startswith('-')
        ]

setup(
    name="llm-eval-pipeline",
    version="0.1.0",
    author="LLM Evaluation Team",
    author_email="eval@example.com",
    description="A comprehensive LLM evaluation pipeline integrating multiple safety, security, and reliability frameworks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/llm-eval-pipeline",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Testing",
        "Topic :: Security",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "nlp": [
            "nltk>=3.8.0",
            "rouge-score>=0.1.2",
            "bert-score>=0.3.13",
        ],
        "distributed": [
            "ray>=2.0.0",
        ],
        "monitoring": [
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
        ],
        "advanced-viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "ml": [
            "scipy>=1.10.0",
            "scikit-learn>=1.3.0",
        ],
        "validation": [
            "pydantic>=2.0.0",
            "jsonschema>=4.17.0",
        ],
        "all": [
            "nltk>=3.8.0",
            "rouge-score>=0.1.2", 
            "bert-score>=0.3.13",
            "ray>=2.0.0",
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "scipy>=1.10.0",
            "scikit-learn>=1.3.0",
            "pydantic>=2.0.0",
            "jsonschema>=4.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-eval=llm_eval.cli:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "llm_eval": [
            "configs/*.yaml",
            "data/*.json",
            "data/*.csv",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/example/llm-eval-pipeline/issues",
        "Source": "https://github.com/example/llm-eval-pipeline",
        "Documentation": "https://llm-eval-pipeline.readthedocs.io/",
    },
    keywords=[
        "llm", "evaluation", "safety", "security", "reliability", 
        "ai", "machine-learning", "testing", "benchmarking"
    ],
)