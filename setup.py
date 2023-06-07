from setuptools import setup, find_packages
import jove

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
                'numpy',
                'scipy',
                'matplotlib',
                'ducc0',
                'ipython',
                'dask[distributed]',
                'pytest >= 6.2.2',
                'pyscilog >= 0.1.2',
                'Click',
                'omegaconf',
                "stimela==2.0rc4",
                'aplpy',
                'xarray',

            ]


setup(
     name='Jovywood',
     version=jove.__version__,
     author="Landman Bester",
     author_email="lbester@ska.ac.za",
     description="Jove related utils",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/landmanbester/Jovywood",
     packages=find_packages(),
     install_requires=requirements,
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     entry_points='''
                    [console_scripts]
                    smoovie=jove.smoovie:cli
                    dspec=jove.dspec:cli

     '''
     ,
 )
