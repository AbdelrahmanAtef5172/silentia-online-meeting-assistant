from setuptools import setup, find_packages

setup(
    name="llm_component",
    version="1.0.0",
    description="LLM-based grammar and vocabulary correction for SLR pipeline output",
    packages=find_packages(exclude=["tests*", "scripts*", "examples*"]),
    python_requires=">=3.10,<3.12",
    install_requires=[
        "httpx>=0.27.0",
        "PyYAML>=6.0",
        "python-dotenv>=1.0",
        "cachetools>=5.3",
    ],
    entry_points={
        "console_scripts": [
            "llm-correct=scripts.run_standalone:main",
            "llm-benchmark=scripts.benchmark:main",
            "llm-check-providers=scripts.test_providers:main",
        ]
    },
)
