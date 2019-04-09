This is an ongoing research around coilguns.
I provide a simple modelisation of the problem using only a few hypothesis:
- Simple RLC circuit, (good enough for damped, SCR triggered coilguns, anyways it is probably hard to have an oscillating circuit in real life)
- I neglect all Eddy currents (just use a laminated projectile, easy... no ?)
- I suppose the quasi-steady state approximation true (this model is probably really wrong for high power coilguns)

# Disclaimer

I decline all responsibilities in case of any accidents. If you are stupid enough to try and build a coilgun out of a random stranger repo on GitHub, it's on you. Plus, it is probably completely illegal where you live.
Come on, I am not even a physicist.

# Installation

To run this code you need to:
- Understand what a coilgun is and how it works (https://en.wikipedia.org/wiki/Coilgun)
- Understand the underlying mathematical model developped in the paper (https://github.com/faameunier/coilgun/blob/master/paper.pdf)
- Install FEMM (http://www.femm.info)
- Install the requirements
`pip install -r requirements.txt`

# How to use 

As this is a pure engineering research, the code is not really user-friendly.

You should define the coilgun setups you want to try in **datastore.py** via the **populate_something** functions. All dimensions are in millimeters, electric tension is in Volt, capacity in Farad, resistance in Ohm.

Then you can just use **main.py** to (in this order):
- Compute some coils data (build_some_coils)
- Compute some optimal launch position for a given electrical setup (build_some_solutions)
- Plot the 3D surface determined by the optimal launch position and have a look at the best coil given a fixed projectile and electrical setup.
- You can then check the impact of a varying magnetic susceptibility if need be.

Please note that running this code can take several days if you want to try a lot of setups on a normal computer.

# Motivation

The mathemical model developped was presented to a jury for the entrance examination for the best French Engineering Schools and was awarded the best possible grade.
The code was originally in Lua and Maple. However, a FEMM-Python connector was recently created and I therefore decided to give this project a second chance. The associated paper might be published one day.
