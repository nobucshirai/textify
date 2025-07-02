# setup.py

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="textify",
    version="0.1.1",
    author="Nobu C. Shirai",
    author_email="5637009+nobucshirai@users.noreply.github.com",
    description="Batch processing audio, video, and document files with Whisper and EasyOCR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nobucshirai/textify",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "openai-whisper @ git+https://github.com/openai/whisper.git",
        "psutil",
        "nvidia-ml-py3",
        "easyocr>=1.7.0",
        "PyMuPDF>=1.22.0",
        "pillow>=10.0.0",
        "watchdog>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "textify=textify.core:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
