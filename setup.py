"""
CPath-Omni: Cross-Species Pathology Foundation Model

Setup script for package installation.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip() 
        for line in requirements_path.read_text().split("\n")
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "open_clip_torch>=2.20.0",
        "Pillow>=9.0.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.60.0",
    ]

setup(
    name="cpath-omni",
    version="1.0.0",
    author="[Author Name]",
    author_email="[email@example.com]",
    description="Cross-Species Pathology Foundation Model for Zero-Shot Cancer Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/[username]/cpath-omni",
    project_urls={
        "Bug Reports": "https://github.com/[username]/cpath-omni/issues",
        "Source": "https://github.com/[username]/cpath-omni",
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
        "wsi": [
            "openslide-python>=1.2.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cpath-inference=scripts.run_inference:main",
            "cpath-demo=scripts.demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml"],
    },
    keywords=[
        "pathology",
        "deep learning", 
        "foundation model",
        "cancer detection",
        "cross-species",
        "zero-shot",
        "CLIP",
        "vision-language",
    ],
)
