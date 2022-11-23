Since _winterdrp_ is an open-source software project, *anyone* can contribute it. You simply need to create a fork of the repository, commit your changes, and then make a pull request.

We have a few general guidelines that are helpful for keeping things organised:

* *Use Github Pull Requests* We like to make sure that the code stays working. So if you are developing something, create a fork of the repo and open a branch. Develop away, and when you are ready, open a pull request. We can then review the code and approve the PR to merge your changes into the main codebase. 

* *Use Github Issues* to coordinate your development. Whether you found a bug, you want to request an enhancement, or you're actively developing a new feature, Github Issues is a great place to keep everyone informed about what you're working on. Click on the label button to provide more info about your topic. Every time you make a relevant PR, remember to tag the issue (e.g `git commit -m 'progress on #12'`), and when you finish and issue you can close it with a commit too! (e.g `git commit -m 'Close #12`').

* *Keep Github Actions Happy!* Github Actions runs all of our unit tests, to make sure you didn't break anything with your commit. You can see if the CI is happy by checking on the github page (look for the badge [![CI](https://github.com/winter-telescope/winterdrp/actions/workflows/continous_integration.yml/badge.svg)](https://github.com/winter-telescope/winterdrp/actions/workflows/continous_integration.yml), or a tick/cross next to your commit). If your commit failed, be sure to check the Github Actions website logs, to see exactly what went wrong.

* *Keep Github Actions Busy!* Github Actions will only run unit tests if we make the unit tests first. When you add a new feature, you also need to add some unit tests so that we can ensure this feature continues to work in the future. Your tests should be saved in the `tests/` directory, and you can find plenty of examples there to copy. Coveralls.io checks how much of the code is covered by tests, and helps you see which lines still need to be covered. You can see all of this on the website: https://coveralls.io/repos/github/winter-telescope/winterdrp or a summary badge [![Coverage Status](https://coveralls.io/repos/github/winter-telescope/winterdrp/badge.svg?branch=main)](https://coveralls.io/github/winter-telescope/winterdrp?branch=main). If your commit adds a lot of new code but does not add unit tests, your commit will be tagged on github with a red cross to let you know that the code coverage is decreasing. If you want to know more about how to design unit tests, you can check out a guide [here](https://medium.com/swlh/introduction-to-unit-testing-in-python-using-unittest-framework-6faa06cc3ee1).

* *Keep the code well-documented* When you write code, it is easier to understamd 'what' than 'why'. People are not mind-readers, and this includes your future self. This is where documentation helps. If you add doctstrings following the [standard python style](https://peps.python.org/pep-0287/), the code can be automatically converted to documentation.

## Updating the documentation

The documentation (generated primarily from docstrings) can be modified with the following command, executed from the docs directory:

```bash
sphinx-apidoc -o source/ ../winterdrp --module-first --force
```

## Checking the tests locally

You can run the tests with:

```TESTDATA_CHECK="True" python -m unittest discover tests/```

This will check that the correct test data version is available, and then run all the tests.

You can also check the code contained within the docstrings/documentation:

```bash
poetry run make -C docs/ doctest
```