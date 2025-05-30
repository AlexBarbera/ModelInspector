from setuptools import setup

setup(
    name="modelinspector",
    version="alpha",
    description="Interactively inspect ML models hidden states.",
    url="",
    author="Alex Barbera",
    license="MIT",
    packages=["modelinspector"],
    requires=[
        "dash",
        "torchvision",
        "plotly",
        "pandas",
        "scipy"
    ],
)