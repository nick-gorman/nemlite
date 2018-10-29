import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nemlite",
    version="0.0.4",
    author="Nicholas Gorman",
    author_email="n.gorman305@gmail.com",
    description="A tool for replicating the NEMDE dispatch procedure.",
    long_description="A tool for replicating the NEMDE dispatch procedure.",
    long_description_content_type="text/markdown",
    url="https://github.com/UNSW-CEEM/nemlite",
    packages=setuptools.find_packages(),
    install_requires=['PuLP', 'joblib', 'pandas', 'osdan'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
)