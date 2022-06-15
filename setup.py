import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

pkgs = {
    "required": [
        "numpy",
        "pandas",
        "torch==1.8.0",
        "pytest",
        "rl4uc",
        "scipy"
    ]
}

setuptools.setup(
    name="ts4uc", 
    version="0.0.1",
    author="Patrick de Mars",
    author_email="pwdemars@gmail.com",
    description="Tree search and reinforcement learning for the unit commitment problem",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=pkgs["required"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9'
)

