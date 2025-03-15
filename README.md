
<div align="center">
        <picture>
                <source media="(prefers-color-scheme: dark)" srcset="./docs/assets/logo.png">
                <source media="(prefers-color-scheme: light)" srcset="./docs/assets/logo_night.png">
                <img alt="Library Banner" src="./docs/assets/logo.png">
        </picture>
</div>
<br>

<div align="center">
    <a href="https://github.com/deel-ai/deel-lip/actions/workflows/python-tests.yml">
        <img alt="Tox" src="https://github.com/deel-ai/deel-lip/actions/workflows/python-linters.yml/badge.svg?branch=master">
    </a>
    <a href="https://pypi.org/project/deel-lip">
        <img alt="Pypi" src="https://img.shields.io/pypi/v/deel-lip.svg">
    </a>
    <a href="https://pepy.tech/project/deel-lip">
        <img alt="Pepy" src="https://pepy.tech/badge/deel-lip">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MIT-efefef">
    </a>
    <br>
    <a href="https://ducoffeM.github.io/keras_custom/"><strong>Explore KerasCustom docs »</strong></a>
</div>
<br>

## 👋 Welcome to keras custom documentation!
***Keras Custom*** is a Python library built qui contient des classes non natives à Keras et des modules de manipulation des modèles Keras. It provides a clear and modular framework on top of Keras, making it a valuable tool for researchers and educators in deep learning.
It enables researchers and developers to efficiently analyze, manipulate, and utilize Keras neural models in downstream tasks.


## 📚 Table of contents

- [📚 Table of contents](#-table-of-contents)
- [🔥 Tutorials](#-tutorials)
- [🚀 Quick Start](#-quick-start)
- [📦 What's Included](#-whats-included)
- [👍 Contributing](#-contributing)
- [🙏 Acknowledgments](#-acknowledgments)
- [📝 License](#-license)

## 🚀 Quick Start

You can install ``keras custom`` directly from pypi:

```python
pip install kerascustom
```

In order to use ``keras custom``, you also need a [valid Keras
installation](https://keras.io/getting_started/). ``keras custom``
supports Keras versions 3.x.

## 🔥 Tutorials

| **Tutorial Name**           | Notebook                                                                                                                                                           |
| :-------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Model splitting - Splitting an existing models into a sequence of nested models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ducoffeM/jacobinet/blob/main/tutorials/PlottingBackward.ipynb)            |
| Model fusion - Combining a sequence of models into a single model with only Layers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ducoffeM/jacobinet/blob/main/tutorials/CustomOp.ipynb)


Documentation is available [**online**](https://ducoffeM.github.io/keras_custom/index.html).



## 👍 Contributing

#To contribute, you can open an
#[issue](https://github.com/ducoffeM/keras_custom/issues), or fork this
#repository and then submit changes through a
#[pull-request](https://github.com/ducoffeM/keras_custom/pulls).
We use [black](https://pypi.org/project/black/) to format the code and follow PEP-8 convention.
To check that your code will pass the lint-checks, you can run:

```python
tox -e py36-lint
```

You need [`tox`](https://tox.readthedocs.io/en/latest/) in order to
run this. You can install it via `pip`:

```python
pip install tox
```


## 🙏 Acknowledgments

<div align="right">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://share.deel.ai/apps/theming/image/logo?useSvg=1&v=10"  width="25%" align="right">
    <source media="(prefers-color-scheme: light)" srcset="https://www.deel.ai/wp-content/uploads/2021/05/logo-DEEL.png"  width="25%" align="right">
    <img alt="DEEL Logo" src="https://www.deel.ai/wp-content/uploads/2021/05/logo-DEEL.png" width="25%" align="right">
  </picture>
  <picture>
    <img alt="ANITI Logo" src="https://aniti.univ-toulouse.fr/wp-content/uploads/2023/06/Capture-decran-2023-06-26-a-09.59.26-1.png" width="25%" align="right">
  </picture>
</div>
This project received funding from the French program within the <a href="https://aniti.univ-toulouse.fr/">Artificial and Natural Intelligence Toulouse Institute (ANITI)</a>. The authors gratefully acknowledge the support of the <a href="https://www.deel.ai/"> DEEL </a> project.


## 🗞️ Citation



## 📝 License

The package is released under <a href="https://choosealicense.com/licenses/mit"> MIT license</a>.




