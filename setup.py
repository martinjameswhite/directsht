import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="directsht",
    version="0.1",
    author="Martin White",
    author_email="mwhite@berkeley.edu",
    description="Code for direct harmonic transforms of scalar fields",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/martinjameswhite/directsht",
    packages=['sht'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy','scipy','numba'],
)
