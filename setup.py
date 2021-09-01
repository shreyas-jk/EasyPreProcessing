from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: 1 - Planning',
  'Intended Audience :: Developers',
  'Operating System :: Microsoft :: Windows :: Windows 10',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
]
 
setup(
  name='easypreprocessing',
  version='1.0.0',
  description='An easy to use pre-processing utility for machine learning.',
  long_description=open('README').read() + '\n\n' + open('CHANGELOG.txt').read(),
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