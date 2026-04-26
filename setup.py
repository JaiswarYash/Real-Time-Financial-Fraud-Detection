from setuptools import setup, find_packages
from typing import List

hypen_e_dot = "-e ."
def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot)
    return requirements

setup(
    name="Fraud_Detection",
    version="0.0.1",
    description='A real-time financial fraud detection system using machine learning.',
    author='Yash Kumar',
    author_email="yashkumr056@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)