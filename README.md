A python module for analysis and visualization of complex-valued "IQ" waveforms.

This is early in development and APIs may change rapidly.

### Install as a module
This makes the `iqwaveform` module available for import from your own python scripts.

#### anaconda distributions
The idea here is to [re-use pre-existing base libraries when possible](https://www.anaconda.com/blog/using-pip-in-a-conda-environment) to minimize interaction problems between pip and conda or mamba.

```python
pip install --upgrade-strategy only-if-needed git+https://github.com/usnistgov/iqwaveform
```

#### other python distributions
```python
pip install git+https://github.com/usnistgov/iqwaveform
```

### Setting up the development environment to run notebooks or develop iqwaveform
The following apply if you'd like to clone the project to develop the module.

1. Clone this repository:

   ```bash
   git clone https://github.com/usnistgov/iqwaveform
   ```

2. Environment setup:
   - Make sure you've installed python 3.8 or newer making sure to include `pip` for base package management (for example, with `conda` or `miniconda`)
   - Make sure you've installed `pdm`, which is used for dependency management isolated from other projects. This only needs to be done once in your python environment (not once per project). To do so, run the following in a command line environment:

      ```bash
      pip install pdm
      ```

      _At this point, close and reopen open a new one to apply updated command line variables_
   - Install the project environment with `pdm`:

      ```bash
      pdm use      
      pdm install
      ```