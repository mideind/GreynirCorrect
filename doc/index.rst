.. GreynirCorrect documentation master file, created by
   sphinx-quickstart on Sun Apr  8 01:20:08 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to GreynirCorrect
=========================

*Til að gagnast sem flestum er skjölun GreynirCorrect á ensku. - In order to serve
the widest possible audience, GreynirCorrect's documentation is in English.*

GreynirCorrect is a Python 3.x package for **correcting spelling and grammar
in Icelandic text**. It relies on the `Greynir <https://pypi.org/project/reynir/>`_
package, by the same authors, to tokenize and parse text.

Here is an example of what GreynirCorrect can do:

.. code-block:: python

   s = check_single("Ég dreimi um að leita af mindinni")
   for a in s.annotations:
      print(a)

Output::

   000-000: P_WRONG_CASE_nf_þf Á líklega að vera 'Mig'
   001-001: S004   Orðið 'dreimi' var leiðrétt í 'dreymi'
   004-005: P001   'leita af' á sennilega að vera 'leita að'
   006-006: S004   Orðið 'mindinni' var leiðrétt í 'myndinni'

It can also be invoked from the command line:

.. code-block:: bash

   $ echo "í auknu mæli fékk Danski maðurin vilja sýnum framgengt að mörgu leiti" | correct
   Í auknum mæli fékk danski maðurinn vilja sínum framgengt að mörgu leyti

You can use this software to implement spelling and grammar checking for
a variety of use cases, including spell-checking of user input in websites,
analyzing existing documents of various kinds, and adding automated proofreading
to text-writing workflows at media companies, law firms, government agencies,
etc.

To get acquainted with GreynirCorrect, we recommend that you start with
the :ref:`overview` and then proceed with the :ref:`installation` instructions.
For further reference, consult the :ref:`reference` section.

This documentation also contains :ref:`information about copyright
and licensing <copyright>`.

Batteries included
------------------

To install and start using GreynirCorrect with Python - and/or as
a :ref:`command-line tool <commandline>` -
you (usually) need :ref:`only one command <installation>`:

.. code-block:: bash

   $ pip install reynir-correct

There is no database to set up or other external dependencies to install;
everything you need is included. GreynirCorrect is thoroughly documented,
and its source code is of course open and
`available on GitHub <https://github.com/mideind/GreynirCorrect>`_.
Your contribution, for instance via pull requests, is welcome!

About the Greynir project
-------------------------

GreynirCorrect is a part of the *Greynir* project,
initiated and maintained by Miðeind ehf. of Reykjavík, Iceland.
It is a free open source software initiative,
started in mid-2015 by its original author, Vilhjálmur Þorsteinsson.
Its aim is to produce an **industrial-strength Natural Language**
**Processing toolset for Icelandic**, with the hope of supporting the
language on the digital front in times of rapid advances in language
technology; changes that may leave low-resource languages at a
disadvantage unless explicit action is taken to strengthen their position.

The Greynir and GreynirCorrect projects are supported by the
Icelandic Government's 5-year Language Technology Programme, and have
in the past benefited from grants from the
`Language Technology Fund (Máltæknisjóður) <https://www.rannis.is/sjodir/rannsoknir/maltaeknisjodur>`__
administered by `Rannís <https://rannis.is>`__.


.. toctree::
   :maxdepth: 1
   :hidden:

   overview
   installation
   commandline
   reference
   copyright

