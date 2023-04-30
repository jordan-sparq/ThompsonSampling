from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Thompson Sampling Package'
LONG_DESCRIPTION = 'This is a python package that deploys Thompson Sampling. This package aims to address real-life ' \
                   'applications and challenges. '

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="ThompsonSampling",
    version=VERSION,
    author="Jordan Palmer",
    author_email="jordan.palmer@datasparq.ai",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],  # add any additional packages that
    # needs to be installed along with your package. Eg: 'caer'

    keywords=['python', 'first package'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)