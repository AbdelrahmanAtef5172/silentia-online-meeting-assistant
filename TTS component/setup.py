from setuptools import setup, find_packages

setup(
    name="tts_component",
    version="1.0.0",
    description="Gender-matched text-to-speech component for the multi-modal pipeline",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.10,<3.15",
    install_requires=[
        "TTS>=0.22.0",
        "edge-tts>=6.1.9",
        "pyttsx3>=2.90",
        "soundfile>=0.12.1",
        "sounddevice>=0.4.6",
        "httpx>=0.27.0",
        "PyYAML>=6.0",
        "python-dotenv>=1.0",
    ],
    entry_points={
        "console_scripts": [
            "tts-speak=cli.run_standalone:main",
            "tts-voices=cli.list_voices:main",
            "tts-benchmark=cli.benchmark:main",
        ]
    },
)
