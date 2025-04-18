import setuptools
from typing import List

def get_requirements(file_path: str) -> List:
    '''
        This function returns a list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if("-e ." in requirements):
            requirements.remove("-e .")
    return requirements



__version__ = "0.0.0"
NAME = "Chicken-Disease-Classification"
USER_NAME = "Mayank"
AUTHOR_EMAIL = "mayankgupta0875@gmail.com"


setuptools.setup(
    name=NAME,
    version=__version__,
    author=USER_NAME,
    author_email=AUTHOR_EMAIL,
    install_requires = get_requirements('requirements.txt'),
    packages=setuptools.find_packages()
)