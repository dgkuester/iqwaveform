# This software was developed by employees of the National Institute of
# Standards and Technology (NIST), an agency of the Federal Government.
# Pursuant to title 17 United States Code Section 105, works of NIST employees
# are not subject to copyright protection in the United States and are
# considered to be in the public domain. Permission to freely use, copy,
# modify, and distribute this software and its documentation without fee is
# hereby granted, provided that this notice and disclaimer of warranty appears
# in all copies.
#
# THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER
# EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY
# THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND FREEDOM FROM
# INFRINGEMENT, AND ANY WARRANTY THAT THE DOCUMENTATION WILL CONFORM TO THE
# SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE ERROR FREE. IN NO EVENT
# SHALL NIST BE LIABLE FOR ANY DAMAGES, INCLUDING, BUT NOT LIMITED TO, DIRECT,
# INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM,
# OR IN ANY WAY CONNECTED WITH THIS SOFTWARE, WHETHER OR NOT BASED UPON
# WARRANTY, CONTRACT, TORT, OR OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED
# BY PERSONS OR PROPERTY OR OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED
# FROM, OR AROSE OUT OF THE RESULTS OF, OR USE OF, THE SOFTWARE OR SERVICES
# PROVIDED HEREUNDER. Distributions of NIST software should also include
# copyright and licensing statements of any third-party software that are
# legally bundled with the code in compliance with the conditions of those
# licenses.

longdescription = """ A python module for analysis and visualization of complex-valued waveforms.
"""

if __name__ == "__main__":
    from distutils.core import setup, Extension
    import platform
    import setuptools
    import sys
    from glob import glob

    sys.path.insert(0, "./iqwaveform")
    from _version import __version__

    PLATFORM_REQUIRES_EXTRAS = []
    PLATFORM_OPTIONAL_EXTRAS = []

    py_version_req = (3, 7)
    if sys.version_info < py_version_req:
        raise ValueError(
            f"python version is {sys.version} but install requires >={'.'.join([str(v) for v in py_version_req])}"
        )

    setup(
        name="iqwaveform",
        version=__version__,
        description="iq waveform analysis and visualization",
        author="Dan Kuester et al",
        author_email="daniel.kuester@nist.gov",
        url="https://github.com/usnistgov/iqwaveform",
        packages=setuptools.find_packages(),
        package_data=dict(
            # these type stubs provide clean call signatures for IDEs
            labbench=["*.pyi", "py.typed"],
        ),
        license="NIST",
        install_requires=[
            # TODO: tighten these requirements a little - perhaps
            # specify ==major version instead of >=
            "pandas>=1.1",
            "scipy",
            "numpy",
            "seaborn",
            "labbench"
        ]
        + PLATFORM_REQUIRES_EXTRAS,
        # entry_points=dict(
        #     # The presence of labbench.__main__ already allows CLI execution with:
        #     #   `python -m labbench`
        #     # The following installs to a scripts directory so that CLI can be accessed
        #     # in the console path as simply
        #     #   `labbench`
        #     # (for supported distributions).
        #     console_scripts=["labbench=cli.__main__:do_cli"]
        # ),
        extras_require=dict(
            notebook=[
                # optional (for now) to reduce dependencies
                # on embedded platforms
                "notebook",
                "ipywidgets",
            ],
        ),
        long_description=longdescription,
        long_description_content_type="text/markdown",
    )
