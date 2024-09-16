'''Module setup file created by PeterC - 30/06/2024'''
from setuptools import setup, find_packages

setup(
    name='pcTorchTools',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "torch==2.2.1",
        "torch-tb-profiler==0.4.3",
        "torchaudio==2.2.1",
        "torchvision==0.17.1",
        "scikit-learn==1.4.1.post1",
        "scipy==1.12.0",
        "numpy==1.26.4",
        "onnx==1.16.1",
        "onnxscript==0.1.0.dev20240609",
        "psutil==5.9.8",
        "tensorboard==2.16.2",
        "tensorboard-data-server==0.7.2",
        "optuna==3.6.1",
        "mlflow==2.14.1",
    ],
    entry_points={
        'console_scripts': [
            # Define command-line executables here if needed
        ],
    },
    author='Pietro Califano',
    author_email='petercalifano.gs@gmail.com',
    description='Custom utility codes based on ray PyTorch for management of DNN (including SNNs) training, validation, logging and conversion. mlflow and optuna are integrated',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='',
    classifiers=[
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
