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


Contributor Guide
-----------------

This guide acts as the single entry point for contributors. It covers coding standards, documentation rules, testing procedures, and the workflow for submitting changes.

Coding Style and Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**PEP 8 Style and Formatting**
    All Python code must adhere to the `PEP 8 style guide <https://peps.python.org/pep-0008/>`_.
    We recommend using tools to ensure compliance:

    - **flake8**: Run ``$ flake8 eelbrain`` from the project root to check for issues locally.
    - **autopep8**: Use this to automatically fix common style issues.
    - **Import Order**: Imports should be grouped in the following order: standard library, third-party libraries (e.g., numpy, matplotlib), and local Eelbrain imports.

**Consistent Naming and API Consistency**
    To make the library intuitive, we strive for consistency across the API:

    - **Naming**: Parameter names should be consistent with existing functions (e.g., use ``cmap`` for colormaps, not ``colorscale``).
    - **API Design**: Functions that handle data should accept data directly as a parameter (e.g., a ``y`` parameter for an ``NDVar``), analogous to existing plotting functions.

**Type Hinting**
    Use type hints in all function signatures (e.g., ``def my_function(y: NDVar) -> Figure:``).
    Do **not** duplicate type information in the docstrings if it is already present in the signature. The signature is the source of truth.

**Docstrings**
    - All docstrings must follow the `numpydoc style <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_.
    - **Class Documentation**: Documentation for a class's ``__init__`` parameters must live in the **class docstring**, not inside the ``__init__`` method itself.

**Documentation Format**
    The documentation is written in `ReStructured Text <https://www.sphinx-doc.org/en/master/usage/restructuredtext>`_.


Recommended Tools
^^^^^^^^^^^^^^^^^

While not required, the following tools are used by the maintainers and can streamline development:

- **SourceTree**: A graphical frontend for git (`link <https://www.sourcetreeapp.com>`_).
- **PyCharm**: A powerful Python IDE that can handle formatting and testing (`link <https://www.jetbrains.com/pycharm>`_).


Testing and Validation
^^^^^^^^^^^^^^^^^^^^^^

Tests for individual modules are included in folders called ``tests``, usually
on the same level as the module.
To run all tests, run ``$ make test`` from the Eelbrain project directory.
On macOS, tests needs to run with the framework build of Python;
if you get a corresponding error, run ``$ ./fix-bin pytest`` from the
``Eelbrain`` repository root.

**Validation Workflow**
    1. **Run tests locally first**: Ensure ``make test`` passes on your machine.
    2. **CI Automation**: All pull requests trigger a Continuous Integration (CI) workflow that automatically runs the full test suite. PRs with failing tests will not be merged.

Opening Issues
^^^^^^^^^^^^^^

Bug reports and feature requests are welcome on the `GitHub Issue Tracker <https://github.com/Eelbrain/Eelbrain/issues>`_.

**Reporting Bugs**
    Effective bug reports help us fix issues faster. Please include:

    - **Minimal Reproducible Example**: A short, self-contained code snippet that demonstrates the error. This is crucial for verification.
    - **Version Information**: Run ``import eelbrain; print(eelbrain.__version__)`` and include the output.
    - **Traceback**: The complete error traceback text.
    - **Description**: A clear description of the expected behavior versus the actual behavior.

**Feature Requests**
    If you plan to implement a new feature, please **open an issue first** to discuss the design. This ensures your contribution aligns with the project's roadmap and avoids wasted effort.

Pull Request Workflow
^^^^^^^^^^^^^^^^^^^^^

We follow a standard Git workflow. For more details, see `GitHub's Pull Request documentation <https://docs.github.com/en/pull-requests>`_.

1. **Create a Branch**: Always create a new branch from ``main`` for your feature or fix.
2. **Commit Changes**: Make your changes and commit them.
3. **Open a Pull Request (PR)**:
    - **Use Draft Mode**: If your work is still in progress, please open the PR in **Draft Mode**. This serves two purposes: it prevents accidental merges while the code is unstable, and it invites maintainers to provide early feedback on your design approach before you invest time polishing the code.
    - **Ready for Review**: Only switch the PR status to "Ready for Review" when you have finished the implementation and verified that local tests pass.
    - Link to any relevant issues or discussions in the PR description.
4. **Keep PRs Small**: Try to keep PRs small and focused on a single issue. Large, monolithic PRs are difficult to review and merge.



Code Review Process
^^^^^^^^^^^^^^^^^^^

The code review process is a collaborative effort to improve code quality.

**expectations**
    A core developer will review your code for correctness, style, and API consistency. Apply feedback globally; if a reviewer notes an issue in one file, check if it exists elsewhere in your changes.

**Responding to Feedback**
    When you have addressed all reviewer comments:
    1. Push the new commits to your branch.
    2. **Leave a comment** on the PR (e.g., "Ready for re-review") or re-request a review via the GitHub UI. This explicitly notifies the maintainers that you are ready for the next round.


Scientific Software Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As a scientific computing library, Eelbrain has strict requirements regarding data integrity:

**Reproducibility**
    All visualizations and computational results must be reproducible across different platforms. Code should produce consistent results given the same input.

**Scientific Accuracy**
    Visualizations must accurately represent the underlying neuroscience data without artifacts. Coordinate systems and data transformations must be mathematically correct.

**Domain Compatibility**
    New features should integrate seamlessly with neuroscience workflows (e.g., support ``NDVar`` objects, work within Jupyter notebooks).

**Performance on Large Datasets**
    Neuroscience datasets are often large. Test your code with realistic data sizes to ensure performance remains acceptable.