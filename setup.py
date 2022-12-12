from setuptools import find_packages,setup
from typing import List

required_file_name = 'requirements.txt'
hyphen_e_dot = '-e .'

def get_requirements()->List[str]:
    with open(required_file_name) as required_file:
        required_list = required_file.readlines()
    required_list = [required_name.replace("\n","") for required_name in required_list]
    if hyphen_e_dot in required_list:
        required_list.remove(hyphen_e_dot)
    return required_list

setup(
    name="sensor",
    version="0.0.1",
    author="Shripad_garat",
    author_email="garatshripad09@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(),)