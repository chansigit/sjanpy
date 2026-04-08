Stacked Bar Plot
================

:func:`~sjanpy.pl.barplot.plot_stacked_bar_repel` creates stacked bar plots
with intelligent label placement for visualizing cell type composition.

Preparing data
--------------

Create a synthetic observation DataFrame:

.. code-block:: python

   import pandas as pd
   import numpy as np

   np.random.seed(42)
   n = 1000
   obs_df = pd.DataFrame({
       'cell_type': np.random.choice(
           ['B cell', 'T cell', 'Monocyte', 'NK', 'DC'],
           size=n, p=[0.3, 0.3, 0.2, 0.1, 0.1]
       ),
       'sample': np.random.choice(
           ['Sample_1', 'Sample_2', 'Sample_3', 'Sample_4'],
           size=n
       ),
   })

Relative composition
--------------------

Show proportions per cell type:

.. code-block:: python

   from sjanpy.pl import plot_stacked_bar_repel

   plot_stacked_bar_repel(
       obs_df,
       group_col='sample',
       type_col='cell_type',
       mode='relative',
       label_content='percentage',
   )

Small slices are automatically labeled with leader lines using ``adjustText``
to avoid overlap.

Absolute counts
---------------

Switch to raw counts with optional log scale:

.. code-block:: python

   plot_stacked_bar_repel(
       obs_df,
       group_col='sample',
       type_col='cell_type',
       mode='absolute',
       log_scale=True,
       label_content='count',
   )

Label options
-------------

The ``label_content`` parameter controls what appears on each bar segment:

- ``'percentage'``: show percentage (default)
- ``'count'``: show raw count
- ``'both'``: show count and percentage

Adjust ``min_label_threshold`` to control when labels switch from inline to
leader-line style (default: 0.03 = 3%).

Saving
------

.. code-block:: python

   plot_stacked_bar_repel(
       obs_df,
       group_col='sample',
       type_col='cell_type',
       save_path='barplot.pdf',
   )
