from setuptools import setup

setup(
    name="ml_utils",
    version="0.1",
    py_modules=["ml_utils"],
    install_requires=["torch", "matplotlib", "numpy"],
    author="Jonas Kleinebecker",
    author_email="jonaskleinebecker@gmail.com",
    description="A simple ML utils package",
    url="https://github.com/JonasKleinebecker/ml_utils",
)
