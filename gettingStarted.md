# Getting Started

The code is this repository is provided in Jupyter notebooks.
In order to use them cleanly, we are going to install a virtual environment, so that you can keep all your python packages neatly contained for this project.
Install python, [pip](https://pip.pypa.io/en/stable/installing/), and [virtualenv](https://virtualenv.pypa.io/en/stable/installation/) if you have not already.

First, clone the git repository:

`git clone https://github.com/desmond-ong/pplAffComp.git`

Within the git repository, make a new virtual environment in a folder name of your choosing. I'll use `env`, but you can use your own:

`cd pplAffComp`

`virtualenv env`

Virtualenv will now install a little python virtual environment. Next, we are going to activate the virtual environment:

`source env/bin/activate`

(You should see `(env)` before your command prompt). To install the required packages:


`pip install -r requirements.txt`

Now you should be able to use Jupyter (and it'll automatically detect the python kernel in the virtualenv). Type:

`jupyter notebook`

And it should open a new browser window. You can then navigate to the various examples to get started!


Once you're done and wish to deactivate the virtualenv, simply type:

`deactivate`

and you should see the `(env)` disappear.


Disclaimer: The current code is written in v0.2.1 of [Pyro](http://pyro.ai/) (which is enforced in the requirements.txt). Pyro is in active development, and there is a chance that future versions of Pyro may not be compatible with this code. We will try our best to keep this working with the latest stable releases of Pyro!
