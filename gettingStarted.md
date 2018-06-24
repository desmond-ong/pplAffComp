# Getting Started

The code in this repository is provided in Jupyter notebooks.
In order to use them, we are going to install a virtual environment, so that you can keep all your python packages neatly contained for this project.
Install python, [pip](https://pip.pypa.io/en/stable/installing/), and [virtualenv](https://virtualenv.pypa.io/en/stable/installation/) if you have not already.

First, open up the command prompt/Terminal and clone the git repository:

`git clone https://github.com/desmond-ong/pplAffComp.git`

Within the git repository, make a new virtual environment in a folder name of your choosing. I'll use `env`, but you can use your own:

```
cd pplAffComp
virtualenv env
```

Virtualenv will now install a little python virtual environment. Next, we are going to activate the virtual environment:

`source env/bin/activate`

You should see `(env)` before your command prompt to indicate that you are working in a virtual environment. 

To install the required packages:

`pip install -r requirements.txt`

--- 
At the current time of writing, the latest release of Pyro is 0.2.1. However, the tutorial uses features in the [/dev branch](https://github.com/uber/pyro) (i.e., the development version of Pyro which may become a later release), so we will be installing Pyro directly from dev.

Make sure you have your environment activated, then type:

```
git clone https://github.com/uber/pyro
cd pyro
pip install .
```

To test if Pyro has been installed properly, start a Python shell by typing:

`python`

And when the python prompt opens, type:

`import pyro`

If it doesn't throw an error, then Pyro has been installed properly. Type `quit()` to return to the command prompt.

---


Now you should be able to use Jupyter (and it'll automatically detect the python kernel in the virtualenv). Type:

`jupyter notebook`

And it should open a new browser window. You can then navigate to the various examples to get started!


Once you're done and wish to deactivate the virtualenv, simply type:

`deactivate`

and you should see the `(env)` disappear.

