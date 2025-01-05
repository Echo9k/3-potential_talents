from setuptools import setup, find_packages
import os
import subprocess
import nltk
import wget
import zipfile


def run_command(command):
    """Run a shell command with error handling."""
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {command}")
        print(e)


def post_install():
    """Post-installation tasks for downloading additional resources."""
    print("Downloading spaCy language model...")
    run_command("python -m spacy download en_core_web_sm")

    print("Installing nbformat...")
    run_command("pip install nbformat")


def download_glove():
    """Download and extract GloVe embeddings."""
    glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"
    glove_zip = "glove.6B.zip"
    if not os.path.exists(glove_zip):
        print(f"Downloading {glove_url}...")
        wget.download(glove_url, glove_zip)
        print("\nDownloaded. Extracting...")
        with zipfile.ZipFile(glove_zip, "r") as zip_ref:
            zip_ref.extractall(".")
        print("GloVe embeddings extracted.")
    else:
        print(f"{glove_zip} already exists. Skipping download.")


def download_nltk():
    """Download required NLTK resources."""
    print("Downloading NLTK resources...")
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download("stopwords")
    nltk.download("averaged_perceptron_tagger")
    print("NLTK resources downloaded.")


if __name__ == "__main__":
    # Perform setup tasks
    download_nltk()
    download_glove()
    post_install()


# Setup configuration
setup(
    name="talent",
    version="1.0",
    description="Setup for talent environment",
    author="Guillermo Alcantara Gonzalez",
    packages=find_packages(where="src"),  # Modern src-layout for Python projects
    package_dir={"": "potential_talents"},
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
