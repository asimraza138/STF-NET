from setuptools import setup, find_packages

setup(
    name="tsfnet",
    version="1.0.0",
    description="Spatio-Temporal Fusion Learning for Robust Deepfake Video Forensics",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "timm>=0.9.12",
        "facenet-pytorch>=2.5.3",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "scikit-image>=0.21.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "PyYAML>=6.0.1",
        "tqdm>=4.66.0",
    ],
    entry_points={
        "console_scripts": [
            "tsfnet-train=scripts.train:main",
            "tsfnet-infer=scripts.inference:main",
            "tsfnet-preprocess=scripts.preprocess_data:main",
        ]
    },
)
