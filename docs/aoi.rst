Areas of Interest
=================

In the ``eyecode`` library, an area of interest (AOI) is a rectangle that has
the following properties:

    * A *kind* or layer
    * A *name*
    * A *bounding box* (x, y, width, height)

Multiple kinds of AOIs can be defined, but AOIs within a single kind (or layer)
should not overlap.  For example, AOIs with a "line" kind and a "token" kind
could be defined for a program. Line AOIs should not overlap with each other,
but they can freely overlap with token AOIs.


Creating AOIs
-------------

AOIs for lines and whitespace-separated tokens can be automatically identified
from a black and white image of the code using the ``find_rectangles``
function.

.. autofunction:: eyecode.aoi.find_rectangles


Hit Testing
-----------

Assigning fixations to AOIs is done using the ``hit_test`` function. As input,
it takes dataframes with fixations and AOIs. The result is a copy of the
fixations dataframe with additional columns for each AOI kind. The value of
each AOI column is the hit AOI name (or NaN if no AOI was hit).

For example, hit testing fixations with AOIs whose kind was "line" and whose
names were "line 1", "line 2", etc. would result in a dataframe with an
"aoi_line" column. The value in this column would be "line 1" when the fixation
hit line 1, "line 2" for line 2, and so on. If no line was hit, the value would
be NaN (pandas default null value).

.. autofunction:: eyecode.aoi.hit_test


Scanpaths
---------

.. autofunction:: eyecode.aoi.scanpath_from_fixations

.. autofunction:: eyecode.aoi.fixations_from_scanpath

.. autofunction:: eyecode.aoi.transition_matrix


Utility Methods
---------------

Below are a few utility functions for making AOI manipulation easier.

.. autofunction:: eyecode.aoi.envelope

.. autofunction:: eyecode.aoi.pad

.. autofunction:: eyecode.aoi.add_bbox

.. autofunction:: eyecode.aoi.get_aoi_columns
