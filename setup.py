from setuptools import setup,find_packages
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(filepath : str) -> List:
    requirements = []

    with open(filepath,"r") as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name = "Flight Price Prediction",
    version = "1.0.0",
    author = "Ankit M Zanzmera",
    author_email = "22msrds052@jainuniversity.ac.in",
    packages= find_packages(where="src"),
    package_dir={"":"src"},
    install_requires = get_requirements('requirements.txt'),
    description= "A project that Predicts the Price of Ticket of Flight by given root"
)