from setuptools import setup

with ("requirements.txt").open(encoding="utf8") as f:
    requirements = f.readlines()

setup(
    name='data_science_framework',
    version='0.0.1',
    description='Data Science Framework',
    url='https://github.com/nogueira-cavalcante/data_science_framework.git',
    install_requires=requirements,
    author='João Paulo Nogueira Cavalcante',
    author_email='jpncavalcante@gmailcom',
    license='unlicense',
    packages=['data_science_framework'],
    zip_safe=False
)