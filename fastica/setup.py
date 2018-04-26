
from setuptools import setup
import setuptools

setup(name = "fastica",
      version = "2.0",
      author='Zhechang Yang and Xi Chen',
      author_email='zhechang.yang@duke.edu',
      url='http://people.duke.edu/~ccc14/sta-663-2018/',
      py_modules = ['fastica'],
      packages=setuptools.find_packages(),
      scripts = ['fastica.py'],
      python_requires='>=3',
      )