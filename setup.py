import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="transferutils",
    version="0.0.2",
    author="Jules Civel, Louis Hache, Younès Rabii, Nathan Trouvain",
    author_email="civel-1@github.com",
    description="Utils for style transfer with TensorFlow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/civel-1/transfer_younes_nathan_jules",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)