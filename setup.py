from setuptools import find_packages, setup

setup(
    name="ml_utils",
    version="0.1",
    packages=find_packages(),
    install_requires=["torch", "matplotlib", "numpy", "tqdm"],
    author="Jonas Kleinebecker",
    author_email="jonaskleinebecker@gmail.com",
    description="A simple ML utils package",
    url="https://github.com/JonasKleinebecker/ml_utils",
)
