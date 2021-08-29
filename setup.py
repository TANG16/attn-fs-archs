import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()    


setuptools.setup(
     name='ndsl-attn-fs-archs',  
     version='0.0.2',
     author="Under ML",
     author_email="urielcoro@gmail.com",
     description="Architectures involved in the investigation of Attention Mechanisms as Feature Selection",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/cobu93/attn-fs-archs",
     package_dir={"": "src"},
     packages=setuptools.find_packages(where="src"),
     classifiers=[
         "Topic :: Scientific/Engineering :: Artificial Intelligence",
         "Development Status :: 4 - Beta" ,
         "License :: Other/Proprietary License",
         "Programming Language :: Python :: 3"
     ],
 )