import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="listener-cbemben",
    version="0.0.1",
    author="cbemben",
    author_email="chrbems@gmail.com",
    description="A set of utilites to ingest and analyze text comments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cbemben/listener",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    include_package_data=True,
    package_data={'': ['data/Reddit_Data.csv']},
)