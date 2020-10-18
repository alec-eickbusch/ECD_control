from setuptools import setup

# TODO: can make this include more information and version control
# TODO: include "analysis" package into setup.
# TODO: move to public repo?
setup(
    name="CD_control",
    version="0.1",
    packages=["CD_control", "analysis"],
    url="https://git.yale.edu/RSL/CD_control",
    license="MIT",
    author="Alec Eickbusch, Shantanu Jha",
    author_email="alec.eickbusch@yale.edu, shantanu.jha@yale.edu",
    description="Conditional Displacement Optimal Control",
)