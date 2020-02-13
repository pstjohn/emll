from setuptools import setup, find_packages

setup(
    name='emll',
    version='0.1.1',
    packages=find_packages(),
    package_data={'emll': ['emll/test_models/*.json',
                           'emll/test_models/*.p',
                           'emll/test_models/*.xml']},
    license="LGPL/GPL v2+",
    long_description=open('README.md').read(),
)
