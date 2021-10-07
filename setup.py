from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()
setup(
    name =          'relearn',
    version =       '0.0.2',  # 0.0.x is for unstable versions
    url =           "https://github.com/Nelson-iitp/relearn",
    author =        "Nelson.S",
    author_email =  "nelson_2121cs07@iitp.ac.in",
    description =   'RL',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages =      ['relearn'],
    license =       'Apache2.0',
    package_dir =   { '' : 'src'},
    #classifiers =   []
    install_requires = ["matplotlib","numpy"],
    #include_package_data=True
)


# ~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+
# BUILD using pip
# ~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+
# cd to dir containing setup.py...
# python setup.py bdist_wheel
# python setup.py sdist
# python setup.py bdist_wheel sdist
# twine upload dist/*