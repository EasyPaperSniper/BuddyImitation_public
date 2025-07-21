from setuptools import setup, find_packages


install_requires = [
    'wandb',
    'einops',
    'tensorboard',
    ]

setup(
    name="TransMimicV2",
    version=1.0,
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False,
)