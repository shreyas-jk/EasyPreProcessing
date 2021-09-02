from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

classifiers = [
  'Development Status :: 1 - Planning',
  'Intended Audience :: Developers',
  'Operating System :: Microsoft :: Windows :: Windows 10',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
]
 
setup(
  name='easypreprocessing',
  version='1.0.3',
  description='An easy to use pre-processing utility for machine learning.',
  long_description=long_description,
  long_description_content_type="text/markdown",
  url='',  
  author='Shreyas Kudav',
  author_email='shreyaskudav@gmail.com',
  license='MIT', 
  classifiers=classifiers,
  keywords='machine learning', 
  packages=find_packages(),
  install_requires=['numpy ','pandas','seaborn ','sklearn','kneed','imbalanced-learn'],
  py_modules=['easypreprocessing'],
  scripts=['easypreprocessing.py'],
  entry_points={
      'console_scripts': ['easypreprocessing=easypreprocessing:main']
  }
)