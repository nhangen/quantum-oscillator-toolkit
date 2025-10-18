from setuptools import setup, find_packages

setup(
    name="quantum-oscillator-toolkit",
    version="0.1.0",
    description="Python toolkit for quantum harmonic oscillator simulation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="@nhangen",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "pytest-cov", "black", "flake8"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)