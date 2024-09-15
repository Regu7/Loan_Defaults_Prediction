from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str) -> List[str]:
    """
    Get the packages from requirements.txt file
    """
    
    with open(file_path,'r') as file_r:
        req = file_r.readlines()
        requirements = [re.replace('\n',"") for re in req]

    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
name='Loan_Default_Prediction',
version='0.0.1',
author='Regu',
author_email='nanthan.regu@gmail.com',
packages=find_packages(),
install_requires=get_requirements(r'deploy\requirements.txt')
)