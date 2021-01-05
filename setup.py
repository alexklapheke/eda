from setuptools import setup, find_packages

setup(
    name="eda",
    version="0.1.2020-01-04",
    description=" Basic EDA tools for data science",
    url="https://github.com/alexklapheke/eda",
    author="Alex Klapheke",
    author_email="alexklapheke@gmail.com",
    license="MIT license",
    packages=find_packages(include=["eda", "eda.*"]),
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "sklearn",
    ]
)
