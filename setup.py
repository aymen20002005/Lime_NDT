from setuptools import setup, find_packages

setup(
    name="Lime_NDT",
    version="0.1.0",
    author="Mohamed Aymen BOUYAHIA",
    author_email="mohamed-aymen.bouyahia@ensta-paris.fr",
    description="Enhancing LIME through Neural Decision Trees",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aymen20002005/Lime_NDT/tree/main",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)