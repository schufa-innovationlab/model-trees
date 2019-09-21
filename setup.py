import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))


def run_setup():
    # Load Version
    with open("VERSION", "r") as f:
        version = f.read().rstrip()

    # Load Requirements
    with open('requirements.txt') as f:
        install_requires = [l.strip() for l in f]

    try:
        readme = open(os.path.join(here, "README.md")).read()
    except IOError:
        readme = ""

    setup(
        name="modeltrees",
        author="schufa-innovationlab",
        version=version,
        description="Scikit-Learn compatible implementation of modeltrees.",
        long_description=readme,
        long_description_content_type="text/markdown",
        license="Apache License 2.0",
        packages=["modeltrees"],
        include_package_data=True,
        url="https://github.com/schufa-innovationlab/model-trees",
        zip_safe=False,
        install_requires=install_requires,
        python_requires=">=3.6",
        classifiers=[
            "Intended Audience :: Science/Research",
            'Intended Audience :: Developers',
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Mathematics",
        ],
    )


if __name__ == "__main__":
    run_setup()