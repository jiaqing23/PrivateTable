import os
from pathlib import Path

if __name__ == '__main__':
    # formating source code
    #os.system('isort *.py **/*.py')
    #os.system('autopep8 --global-config setup.cfg -i *.py **/*.py')
    # For Windows use below
    os.system('isort .')
    os.system('autopep8 --global-config setup.cfg -i -r .')

    # type checking
    #os.system('mypy *.py tests/*.py')
    # For Windows use below
    os.system('mypy .')
    os.system('mypy tests')

    # run unit tests using pytest
    os.system('pytest -s .')

    # clean up
    #os.system('py3clean .')
    os.system('rm -r .pytest_cache')
    # For Windows use below
    os.system('pyclean .')
