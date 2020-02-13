from setuptools import setup, find_packages

setup(
    name='emll',
    version='0.1.1',
    packages=find_packages(),
    include_package_data=True,
    license="LGPL/GPL v2+",
    long_description=open('README.md').read(),
)
