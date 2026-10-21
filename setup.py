from setuptools import setup, find_packages

setup(
    name='realtime-taxi-routing',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/lab-core/realtime-taxi-routing',
    license='',
    author='Elahe Amiri',
    author_email='elahe.amiri@polymtl.ca',
    description='Real-time taxi routing system with multimodal simulator integration.',
    install_requires=[
        'wheel>=0.36.0',
        'setuptools>=50.0.0',
        'pip>=21.0',
        'tabulate',
        'scipy',
        'matplotlib==3.8',
        'seaborn',
        'gurobipy'
    ],
    include_package_data=True,
)