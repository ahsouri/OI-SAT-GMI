from setuptools import setup,find_packages

with open('README.md') as f:
    readme = f.read()

with open('requirements.txt') as fp:
    install_requires = fp.read()

setup(name='OISATGMI',
      version='0.0.1',
      description='Optimal Interpolation Between GMI and a Satellite-based Observations',
      long_description=readme,
      long_description_content_type='text/markdown',
      author='Amir Souri',
      author_email='ahsouri@gmail.com',
      license='MIT',
      packages=['oisatgmi'],
      install_requires=install_requires,
      zip_safe=False)
