.. _development:

***********
Development
***********

Eelbrain is hosted on `GitHub <https://github.com/Eelbrain/Eelbrain>`_.


The Development Version
-----------------------

Development takes place on the ``main`` branch, while release versions are maintained on
branches called ``r/0.26`` etc. For further information on working with
GitHub see `GitHub's instructions <https://help.github.com/articles/fork-a-repo/>`_.

The repository contains a conda environment that contains everything needed to use Eelbrain except Eelbrain itself.
First, clone the repository (or your `fork <https://help.github.com/articles/fork-a-repo>`_), and change into the repository directory::

    $ git clone https://github.com/Eelbrain/Eelbrain.git
    $ cd Eelbrain

To generate the ``eeldev`` environment, use::

    $ mamba env create --file=env-dev.yml

The development version of Eelbrain can then be installed with ``pip``::

    $ mamba activate eeldev
    $ pip install -e .

On macOS, the ``$ eelbrain`` shell script to run ``iPython`` with the framework
build is not installed properly by ``pip``; in order to fix this, run::

    $ ./fix-bin

In Python, you can make sure that you are working with the development version::

    >>> import eelbrain
    >>> eelbrain.__version__
    'dev'


Contributing
------------

Contributions to code and documentation are welcome as pull requests into the ``main`` branch.

Style guides:

- Python code style follows `PEP8 <https://www.python.org/dev/peps/pep-0008>`_ (mostly)
- The documentation is written in `ReStructured Text <https://www.sphinx-doc.org/en/master/usage/restructuredtext>`_
- Docstrings follow the `numpydoc style  <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_


Useful tools:

- Graphical frontend for git: `SourceTree <https://www.sourcetreeapp.com>`_
- Python IDE: `PyCharm <https://www.jetbrains.com/pycharm>`_


Testing
-------

Tests for individual modules are included in folders called ``tests``, usually
on the same level as the module.
To run all tests, run ``$ make test`` from the Eelbrain project directory.
On macOS, tests needs to run with the framework build of Python;
if you get a corresponding error, run ``$ ./fix-bin pytest`` from the
``Eelbrain`` repository root.


Contributor Guide
-----------------

Why a Contributor Guide is Necessary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A contributor guide is a critical document for any open-source software project, particularly in a scientific context like Eelbrain.
Its primary purpose is to standardize the development process and lower the barrier to entry for new contributors.
For a project to thrive and grow, it must be able to attract and effectively integrate new developers, who may range from students to researchers at other institutions.

A well-written guide provides a clear roadmap for everything from setting up a development environment and running tests to following coding conventions and submitting code for review.
This standardization ensures that all contributions maintain a consistent level of quality, making the codebase more coherent and easier to manage in the long term.
By removing ambiguity and providing clear instructions, a contributor guide empowers potential collaborators, reduces the burden on core developers to answer repetitive questions, and fosters a welcoming and productive community.


Coding Conventions
^^^^^^^^^^^^^^^^^^

To maintain a consistent and readable codebase, we adhere to the following conventions, which were established and refined through code review discussions.

**Style Guide**
  All Python code must adhere to the `PEP 8 style guide <https://peps.python.org/pep-0008/>`_.
  We use ``flake8`` to check for compliance. Before submitting your code, please run ``flake8`` locally and fix any reported issues.
  You can use tools like ``autopep8`` to automatically fix many common style issues.

**Import Order**
  Imports should be grouped in the following order, with a blank line separating each group:

  1. Standard library imports (e.g., ``io``, ``os``)
  2. Related third-party imports (e.g., ``numpy``, ``matplotlib``)
  3. Local application/library specific imports (e.g., ``from eelbrain import ...``)

  *Case Study:* In one review, a contributor was reminded to reorder their imports to place standard library modules before third-party and local imports.
  Following this guideline makes it easier to see the dependencies of a module at a glance.

**API Consistency**
  To make the library intuitive, we strive for consistency across the API:

  - Parameter names should be consistent with existing functions. For example, use ``cmap`` for colormaps, not ``colorscale``.
  - Functions that handle data should, where possible, accept data directly as a parameter (e.g., a ``y`` parameter for an ``NDVar``), analogous to existing plotting functions.

  *Case Study:* During the development of the interactive plotting feature, a reviewer noted that a new function used the parameter name ``colorscale``.
  The contributor was asked to rename it to ``cmap`` to match the parameter name used for colormaps throughout the rest of the Eelbrain library.

**Type Hinting**
  Use type hints in all function signatures (e.g., ``def my_function(y: NDVar) -> Figure:``).
  When type hints are present in the signature, they should be omitted from the docstring to avoid redundancy.

**Docstrings**
  All public functions and classes should have clear and informative docstrings.
  Documentation for a class's ``__init__`` method should be included in the main class docstring.

**Data Access**
  When working with ``NDVar`` objects, use the ``.get_data()`` method to ensure the data axes are in the desired order, as the internal order is not guaranteed.

**TODOs**
  For in-code reminders, use a ``TODO:`` tag. For more significant or non-localized tasks, please open a GitHub issue instead.

**Development Tools**
  To streamline the development process and maintain code quality:

  - Use ``flake8`` locally to check code compliance before submitting
  - Consider using ``autopep8`` to automatically fix common style issues
  - Configure your IDE to automatically handle formatting (e.g., PyCharm can manage whitespace issues automatically)
  - Run local checks to catch style problems before they appear in CI

**API Design Principles**
  When designing new functionality:

  - Maintain parameter naming consistency across the codebase (e.g., use ``cmap`` for colormaps, not ``colorscale``)
  - Follow existing patterns for data input (e.g., accept data directly via a ``y`` parameter like other plotting functions)
  - Consider usability in Jupyter environments during design
  - Document expected data formats clearly for users


Architecture and Dependency Guidelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To keep contributions sustainable and reviews effective, follow these project-level guidelines:

- **Reduce complexity via modular boundaries**: keep the Eelbrain core unchanged when possible; place specialized visualization or UI features in separate packages.
- **Isolate dependencies**: avoid mixing incompatible stacks (e.g., Plotly vs. Matplotlib) within one module; separate packages prevent CI failures and version conflicts.
- **Prefer smaller, focused units**: submit smaller repositories/modules and small PRs to simplify review, testing, and independent evolution.


The Pull Request (PR) Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use a pull request-based workflow for all contributions.

1. **Create a branch**: Create a new branch from ``main`` for your feature or bugfix.

2. **Make your changes**: Make your code changes, ensuring you follow the coding conventions.

3. **Submit a Pull Request**: When your changes are ready, push your branch to your fork and open a pull request against the ``main`` branch of the official Eelbrain repository.

   - Provide a clear and descriptive title for your PR.
   - In the description, explain the purpose of your changes and link to any relevant GitHub issues.

4. **Keep PRs small**: Whenever possible, break down large features into smaller, logically distinct pull requests. Small PRs are much easier and faster to review.

   *Case Study:* The initial implementation of the "Live Neuron" feature was submitted as a single large PR (+1,873 lines of code).
   Reviewers noted that this made the review process very challenging.
   This experience highlighted the importance of smaller, more focused PRs to facilitate timely and effective feedback.

5. **Address feedback**: Engage with the code review process by responding to comments and pushing new commits to your branch to address the feedback.


Testing and Validation
^^^^^^^^^^^^^^^^^^^^^^^

Eelbrain relies on a robust test suite to ensure the correctness and reliability of its scientific algorithms.

**Automated Testing**
  All pull requests trigger a Continuous Integration (CI) workflow that automatically runs the full test suite.
  Please ensure all tests are passing before requesting a review.

  *Case Study:* The large "Live Neuron" PR initially failed the automated tests because a new dependency (Plotly) created conflicts with the existing Matplotlib-based code.
  This demonstrated the value of CI in catching integration issues early.
  The eventual solution was to develop the new feature as a separate, independent module to resolve the conflict.

**Writing New Tests**
  Any new feature or bugfix should be accompanied by corresponding unit tests.
  This helps prevent future regressions and validates that your code is working as expected.

**Integration Testing**
  For new modules (especially those with external dependencies like Plotly), ensure they:

  - Are compatible with the Jupyter notebook environment
  - Do not conflict with existing dependencies (particularly Matplotlib)
  - Work across supported platforms (Linux, Windows 10+, macOS)
  - Meet scientific visualization quality standards for accuracy and reproducibility

**Performance Considerations**
  When implementing interactive features:

  - Test with realistic data sizes and ensure reasonable response times
  - Consider memory usage for large datasets
  - Optimize for common use cases (e.g., showing important dipoles rather than all dipoles)
  - Verify that plot sizing works correctly within Jupyter cell outputs


Code Review Process
^^^^^^^^^^^^^^^^^^^

The code review process is a collaborative effort to improve the quality of the codebase.

**What to Expect**
  A core developer will review your pull request and may provide feedback on various aspects, including correctness, adherence to coding standards, API design, and usability.
  The process is iterative; you'll be expected to update your PR based on the feedback.

**Applying Feedback**
  When you receive feedback (e.g., about import order or type hints), please check your entire contribution to see if the same feedback applies elsewhere.
  This helps streamline the review process.

  *Case Study:* A reviewer provided feedback on the import order in one file. Later in the same review, the same issue was found in another file.
  This led to a friendly reminder to contributors to apply feedback globally across their entire PR, which makes the review more efficient for everyone.

**Goal**
  The goal of the review is not just to find errors but to refine the design and ensure the new code is well-integrated into the existing project.
  Once the core technical aspects are settled, we may also seek feedback from other users on the usability of a new feature.

**Common Review Focus Areas**
  Based on actual code review experiences, reviewers typically examine:

  - API consistency with existing functions (parameter names, data input patterns)
  - Code organization and import structure
  - Documentation completeness and clarity (including links to relevant external documentation)
  - Error handling for edge cases (e.g., data without certain dimensions)
  - User experience considerations (layout compactness, interaction patterns)
  - Performance implications for large datasets

**Response to Feedback**
  When addressing reviewer comments:

  - Apply feedback comprehensively across your entire contribution
  - Ask clarifying questions if requirements are unclear
  - Consider creating follow-up PRs for related improvements suggested during review
  - Document any design decisions that may not be immediately obvious


Scientific Software Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As a scientific computing library, Eelbrain has additional requirements beyond typical software projects:

**Reproducibility**
  All visualizations and computational results must be reproducible across different platforms and Python environments.
  Code should produce consistent results given the same input data and parameters.

**Scientific Accuracy**
  Visualizations must accurately represent the underlying neuroscience data without introducing artifacts or misleading interpretations.
  This includes proper handling of coordinate systems, color scales, and data transformations.

**Domain Compatibility**
  New features should integrate seamlessly with the neuroscience workflow:

  - Support standard data formats (NDVar objects with appropriate dimensions)
  - Work within Jupyter notebook environments commonly used by researchers
  - Provide clear documentation of data expectations and output formats
  - Consider the typical use cases of neuroscience researchers

**Performance for Research Data**
  Neuroscience datasets can be large and complex:

  - Test with realistic data sizes (multiple subjects, high temporal resolution)
  - Optimize for common operations (time series visualization, source localization displays)
  - Provide user control over performance trade-offs (e.g., showing all vs. significant dipoles)

By investing in such a guide, the Eelbrain project can significantly improve its collaborative development process, making it more efficient, inclusive, and sustainable.
