# Mag_PINN
A physics informed neural network (PINN) toy model for earth's magnetic field
The satellites grab magnetic field data around the earth. We can train a model of the magnetic field, but the basic physics rule of it is divergence free.
The PINN will add a loss term to penalize the magnetic field's divergence and make the model divergence-free.
An example can be found in the jupyter notebook.


