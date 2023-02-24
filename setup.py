from setuptools import setup,find_packages

with open('README.md') as f:
    readme = f.read()

setup(name='OISATGMI',
      version='0.0.1',
      description='Optimal Interpolation Between GMI and a Satellite-based Observations',
      long_description=readme,
      long_description_content_type='text/markdown',
      author='Amir Souri',
      author_email='ahsouri@gmail.com',
      license='MIT',
      packages=['oisatgmi'],
      install_requires=[
          'numpy','matplotlib','scipy','netCDF4','scikit-gstat'
          ],
      zip_safe=False)
