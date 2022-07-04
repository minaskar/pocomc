import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

try:
    with open("requirements.txt", 'r') as dependencies:
        requirements = [pkg.strip() for pkg in dependencies]
except FileNotFoundError as e:
    print(e)
    print("Using hardcoded requirements")
    requirements = [
        "numpy",
        "torch",
        "tqdm",
        "matplotlib",
        "scipy",
        "corner"
    ]
        
setuptools.setup(
    name="pocomc",
    version="0.0.1",
    author="Minas Karamanis",
    author_email="minaskar@gmail.com",
    description="Preconditioned Monte Carlo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/minaskar/pocomc",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    install_requires=requirements,
    python_requires='>=3.7',
)
