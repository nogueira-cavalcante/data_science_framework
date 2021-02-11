from setuptools import setup

with ("requirements.txt").open(encoding="utf8") as f:
    requirements = f.readlines()

setup(
    name='BRADA',
    version='0.0.1',
    description='My private package from private github repo',
    url='https://github.com/nogueira-cavalcante/BR_ML.git',
    install_requires=requirements,
    author='João Paulo Nogueira Cavalcante',
    author_email='joaocavalcante@br.com.br',
    license='unlicense',
    packages=['BRADA'],
    zip_safe=False
)
