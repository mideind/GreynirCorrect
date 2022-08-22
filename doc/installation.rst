.. _installation:

Installation
============

Prerequisites
-------------

GreynirCorrect runs on **CPython 3.7** or newer, and on **PyPy 3.7**
or newer (more info on PyPy `here <http://pypy.org/>`_).

On GNU/Linux and similar systems, you may need to have ``python3-dev``
installed on your system:

.. code-block:: bash

    # Debian or Ubuntu:
    $ sudo apt-get install python3-dev

Depending on your system, you may also need to install ``libffi-dev``:

.. code-block:: bash

    # Debian or Ubuntu:
    $ sudo apt-get install libffi-dev

On Windows, you may need the latest
`Visual Studio Build Tools <https://www.visualstudio.com/downloads/?q=build+tools+for+visual+studio>`_,
specifically the Visual C++ build tools, installed on your PC along
with the Windows 10 SDK.


Install with pip
----------------

To install GreynirCorrect:

.. code-block:: bash

    $ pip install reynir-correct

...or if you want to be able to edit GreynirCorrect's source code in-place,
install ``git`` and do the following:

.. code-block:: bash

    $ mkdir ~/github
    $ cd ~/github
    $ git clone https://github.com/mideind/GreynirCorrect
    $ cd GreynirCorrect
    $ pip install -e .

On the most common Linux x86_64/amd64 systems, ``pip`` will download and
install a binary wheel. On other systems, a source distribution will be
downloaded and compiled to binary.

Pull requests are welcome in the project's
`GitHub repository <https://github.com/mideind/GreynirCorrect>`_.


Install into a virtualenv
-------------------------

In many cases, you will want to maintain a separate Python environment for
your project that uses GreynirCorrect. For this, you can use *virtualenv*
(if you haven't already, install it with ``pip install virtualenv``):

.. code-block:: bash

    $ virtualenv -p python3 venv

    # Enter the virtual environment
    $ source venv/bin/activate

    # Install GreynirCorrect into it
    $ pip install reynir-correct

    $ python
        [ Use Python with GreynirCorrect ]

    # Leave the virtual environment
    $ deactivate

On Windows:

.. code-block:: batch

    C:\MyProject> virtualenv venv

    REM Enter the virtual environment
    C:\MyProject> venv/Scripts/activate

    REM Install GreynirCorrect into it
    (venv) C:\MyProject> pip install reynir-correct

    (venv) C:\MyProject> python
        REM [ Use Python with GreynirCorrect ]

    REM Leave the virtual environment
    (venv) C:\MyProject> deactivate

More information about *virtualenv* is `available
here <https://virtualenv.pypa.io/en/stable/>`_.
