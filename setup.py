from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = [
    "matplotlib<=3.6.0",
    "numpy",
    "pandas",
    "ppscore",
    "scikit-learn",
    "scipy",
    "statsmodels",
]

setup(
    name="datavizml",
    author="Dr Robert Dibble",
    author_email="robertdibble@live.co.uk",
    description="A package to explore and visualise a dataset in preparation for an ML project",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/dibble07/datavizml",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
