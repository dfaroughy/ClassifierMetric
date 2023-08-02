
import setuptools

def load_requirements():
    try:
        with open("requirements.txt") as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("WARNING: requirements.txt not found")
        return []
try:
    with open("README.md", "r") as f:
        long_description = f.read()
except:
    long_description = "# gen_model_classifier"

setuptools.setup(
    name="graph_bridges",
    version=1.0,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="git@github.com:dfaroughy/ClassifierMetric.git",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"}
)