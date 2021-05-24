from setuptools import setup

# TODO: can make this include more information and version control
# TODO: include "analysis" package into setup.
# TODO: move to public repo?
setup(
    name="CD control optimization",
    version="0.1",
    packages=["ECD_control_optimization", "ECD_pulse_construction"],
    url="https://git.yale.edu/RSL/ECD_control",
    license="MIT",
    author="Alec Eickbusch, Shantanu Jha",
    author_email="alec.eickbusch@yale.edu, shantanu.jha@yale.edu",
    description="Echoed Conditional Displacement Optimal Control",
)
