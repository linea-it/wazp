from setuptools import setup, find_packages
import sys

version = sys.version_info
required_py_version = 3.11
if version[0] < int(required_py_version) or (
    version[0] == int(required_py_version)
    and version[1] < required_py_version - int(required_py_version)
):
    raise SystemError("Minimum supported python version is %.2f" % required_py_version)


# adapted from pip's definition, https://github.com/pypa/pip/blob/master/setup.py
def get_version(rel_path):
    with open(rel_path) as file:
        for line in file:
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                version = line.split(delim)[1]
                return version
    raise RuntimeError("Unable to find version string.")


setup(
    name="wazp",
    version=get_version("wazp/__init__.py"),
    author="Christophe Benoist",
    license="BSD 3-Clause License",
    url="https://github.com/linea-it/wazp/",
    packages=find_packages(),
    description="XXXXX",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD 3-Clause",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    #install_requires=["astropy", "numpy", "scipy", "healpy", "scikit-image", "sc
    install_requires=["scikit-image", "scikit-learn", "astropy", "healpy"],
    #install_requires=["astropy>3.0", "numpy=2.5"],
    python_requires=">" + str(required_py_version),
)
