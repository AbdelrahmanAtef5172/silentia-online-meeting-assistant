from setuptools import setup, find_packages

setup(
    name="gender_detection",
    version="1.0.0",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.10,<3.12",
    install_requires=[
        "torch>=1.13",
        "torchvision>=0.14",
        "timm>=0.6",
        "opencv-python>=4.5",
        "PyYAML>=5.4",
        "python-dotenv>=0.19",
        "numpy>=1.21",
        "Pillow>=9.0",
        "tqdm>=4.60",
    ],
    extras_require={
        "test": ["pytest", "pytest-cov"],
    },
)
