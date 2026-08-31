Plotting
========

:mod:`.plotting`
----------------

.. warning:: **Experimental.** Used by some scripts

.. warning::

  **Matplotlib backend rules**

  - **For library / plotting modules**

    - These modules assume the backend has already been configured.
    - So they import ``matplotlib.pyplot`` normally.

  - **For scripts (CLI / entrypoints)**

    - Configure matplotlib backend before any plotting. You can use
      :func:`sgs_tools.scripts.plotting.configure_matplotlib_backend`
      to configure at runtime or import time.
    - To allow using CLI programmatically, **do not** import
      ``matplotlib.pyplot`` at module level,
      and run `configure_matplotlib_backend` as first thing in `main`.


.. automodule:: sgs_tools.plotting.field_plot_map
   :members:

.. automodule:: sgs_tools.plotting.collection_plots
   :members:

..automodule:: sgs_tools.plotting.handle_figure
   :members:

.. literalinclude:: ../src/sgs_tools/plotting/plot_config_template.json
   :language: json
   :caption: Plot style template file
