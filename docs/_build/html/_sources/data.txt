Experiment Data
===============

The complete dataset from the `eyeCode experiment
<http://eyecode.synesthesiam.com>`_ is available in the ``eyecode.data`` module.

Hansen 2012
-----------

Pandas dataframes with information about the experiment programs, individual
trials, experiments/participants are available for the combined Mechanical Turk
and eye-tracking participant sets.

.. autofunction:: eyecode.data.hansen_2012.programs

.. autofunction:: eyecode.data.hansen_2012.program_code

.. autofunction:: eyecode.data.hansen_2012.program_output

.. autofunction:: eyecode.data.hansen_2012.trials

.. autofunction:: eyecode.data.hansen_2012.experiments

For the eye-tracking experiments (29 out of 162 experiments), fixations, areas
of interest, and screenshots are available. Full trial videos can be seen at
the `eyecode website
<http://eyecode.synesthesiam.com/stories/eye-tracking-videos.html>`_.

.. autofunction:: eyecode.data.hansen_2012.all_fixations

.. autofunction:: eyecode.data.hansen_2012.areas_of_interest

.. autofunction:: eyecode.data.hansen_2012.trial_screen
