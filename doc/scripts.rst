Scripts
===========


Post-Processing
-----------------------

.. sphinx_argparse_cli::
   :module: sgs_tools.scripts.CS_calculation_genmodel
   :func: parse_args
   :hook:
   :prog: cs_dynamic
   :group_title_prefix:
   :title: CS_calculation_genmodel.py

.. sphinx_argparse_cli::
   :module: sgs_tools.scripts.post_process
   :func: parse_args
   :hook:
   :prog: post_process
   :group_title_prefix:
   :title: post_process.py

.. sphinx_argparse_cli::
   :module: sgs_tools.scripts.BasicComparisonSimAnalysis
   :func: parse_args
   :hook:
   :prog: sim_comparison
   :group_title_prefix:
   :title: BasicComparisonSimAnalysis.py

.. sphinx_argparse_cli::
   :module: sgs_tools.scripts.ReferenceComparisonSimAnalysis
   :func: parse_args
   :hook:
   :prog: ref_comparison
   :group_title_prefix:
   :title: ReferenceComparisonSimAnalysis.py


Pre-processing
---------------

.. sphinx_argparse_cli::
   :module: sgs_tools.scripts.make_UM_level_sets
   :func: parser
   :hook:
   :prog: um_levels
   :group_title_prefix:
   :title: make_UM_level_sets.py

Miscellaneous
---------------

.. sphinx_argparse_cli::
   :module: sgs_tools.scripts.version
   :func: main
   :hook:
   :prog: sgs_tools_version
   :group_title_prefix:
   :title: version.py

Utilities:
---------------

.. automodule:: sgs_tools.scripts.arg_parsers
   :members:

.. automodule:: sgs_tools.scripts.cli_helpers
   :members:

.. automodule:: sgs_tools.scripts.plotting
   :members:
