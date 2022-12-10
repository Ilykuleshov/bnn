from pathlib import Path
import runpy

from setuptools import setup, find_packages, find_namespace_packages

# automatically detect the lib name
root = Path(__file__).resolve().parent
dirs = {
    d.parent for d in root.rglob('__version__.py')
}
assert len(dirs) == 1, dirs
folder, = dirs
name = str(folder.relative_to(root)).replace('/', '.')
init = runpy.run_path(folder / '__version__.py')
if name == 'my-project-name':
    raise ValueError('Rename "my-project-name" to your project\'s name.')
if '__version__' not in init:
    raise ValueError('Provide a __version__ in __init__.py')

version = init['__version__']
with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()
with open('README.md', encoding='utf-8') as file:
    long_description = file.read()

find = find_namespace_packages if '.' in name else find_packages

setup(
    name=name,
    packages=find(include=(name,)),
    include_package_data=True,
    version=version,
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    # OPTIONAL: uncomment if needed
    # python_requires='>=3.6',
)
