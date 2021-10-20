---
title: 'Atomic Cluster Environments'
date: October 2021
permalink: /posts/2021/10/ace-descriptors
tags:
  - machine learning
  - molecular dynamics
  - molecular descriptors
---

Introduction to Atomic Descriptors
======
In classical molecular dynamics, the interatomic potential is approximated as a summation of bonded and non-bonded interactions.  Empirical forcefields parametrise these with simple functional forms (1), resulting in few free parameters that can be fitted, either by hand, or automatically, to experimental and ab initio electronic structure calculations.  Whilst these potentials are fast to evaluate and are capture intuitive body-ordered interactions (bonds, angles, torsions), they are fundamentally limited in their accuracy and cannot, without significant modification, capture reactivity or non-equilibrium dynamics.

$$ E = \sum_{bonds} k_r(r-r_{eq})^2 + \sum_{angles} k_{sigma} (\theta - \theta_{eq})^2 + \sum_{dihedrals} \frac{v_n}{2} (1 + cos(n\phi - \gamma)) + E_{non-bonded}$$


An alternative formulation
------
In materials science, another approach predominates - potentials can be fitted to descriptors of the atomic envoironment.  One common approach uses the idea of 'neighbourhood density' to capture the local atomic environment in terms of the interaction of each atom with its immediate neighbours.  By using this as your basis function, you can retain a body-ordered approach without being limited by the fixed functional form of empirical forcefields.

The general approach
-----
Many formulations exist in the materials modelling literature that rely on a similar approach.  Fundamentally, atomic neighbour density is a poor basis function from which to construct a potential - whilst it provides translational and permutational invariance between like atoms 'for free', it fails to capture rotational invariance.  

$$ \rho_{i}^{z}(r) = \sum_j \delta_{z z_j}\delta(r-r_{ij}) $$

Multiple approaches apply the same idea - projection of the atomic density onto a rotationally invariant basis allow us to construct descriptor vectors centered on each atom in the system.

$$ 
A_{z_i znlm} = \langle \rho_i^{z} | \phi_{nlm}^{z_i z} \rangle = \sum_j \phi_{nlm}^{z_i z_j} (r_{ij})
$$

Originally employed in the SOAP descriptors, a basis function containing a product of a radial component (of unspecified functional form) and a spherical harmonics can be integrated over the rotational point group O(3) to provide a rotationally invariant descriptor.  

In both the SOAP and ACE framework, the 'density trick' is employed: essentially the basis set is constructed a product of the projected density functions, provides a computationally easier way to evaluate the cross terms and construct a body-ordered representation, with an evaluation time that scales linearly with basis set size, instead of exponentially.

$$ A_{z_i \mathbf{v}} = \prod_{t=1}^v A_{z_i v_t} $$

The total energy of the system is then a sum of the atomic energies, each of which if computed from a fitted linear model in the basis set functions.

$$ E_i = \sum_v c_{v_i v} A_{z_i v} + \sum_v c_{v_i v_1 v_2} A_{z_i v_1}A_{z_i v_2} + ... $$

Hyperparameters of the ACE framework
-----

In the ACE framework, we have some flexibility over the radial component of the basis function.  It is not entirely clear which functional forms are appropriate for a given problem.

We fit the parameters over the entire set of chemical environments, intuitively there should be a trade off between the size of the dataset you want to fit (number of unique chemical environments), the max body order of the terms to use, and the maximum polynomial degree of the basis funcitons (dependent on the indices of the spherical harmonics)

In the fit we minimise combined MSE for predicted energies and forces summed over the molecules in the configuration (i.e. configurations from an ab-initio MD trajectory)


Results
-----
With an isometrically invariant basis set in hand, we can see how good it is for making predictions on a dataset.  In principle, we could use these descriptors to predict any property we want (SOAP has been sucessfully employed as an alternative to tranditional QSAR descriptors), but for forcefield parametrisation, we want to fit the model to the total energy of the system.  Note that this fits into the interesting class of machine learning problems where we attempt to infer some unobservable value (in this case the per-atom contribution) as a derived property of the total energy of the system (which is what we get from QM).  

In general, fitting a functional form up to v=4 (5 body terms) is about the maximum you can accept before the scaling kicks in and evaluation time per atom becomes too slow.  An interesting question is how to sparsify your basis set to construct an optimal Paretto frontier in terms of accuracy and evaluation time.  Note that ACE is implemented in Julia currently.

ACE models trained on QM data from MD17 are able to accurately predict total forces and energies with MAEs on par or better than state of the art approaches.  Crucially, it is also extremely performant in the low-data limit where few training configurations are provided, relative to the other models, and in every case draws the Pareto frontier in terms of speed vs accuracy with various basis set sizes.  

A key question in modelling drug-like organic molecules is how well the model performs far from the training set geometries, which means you don't have to retrain your potential energy surface at multiple temperatures.  Careful regularisation is key to ensuring a smooth potential here.

An interesting question arrises when you train on the entire dataset.  ACE errors increase by more than the other models (partially because the neural network approaches offer a more flexible functional form, indicating ACE underfits to this data).  Despite this, the overall errors from the ACE model are still lowest, however with additional training set size, this may not be the case.  It does demonstrate some ability to generalise to completely unknown molecules though, and uses orders of magnitude less training data than the neural net approaches.  

Summary
-----
- ACE models use a projection of density onto a rotationally invariant basis set to construct a body-ordered forcefield.
- The 'density trick' allows for the construction of a high body-ordered basis set without the requirement for evaluating each cross-term in the expansion
	- Scaling is not affected by the body order explicitly, but the number of functions does scale exponentially with body order, and overall evaluation time scales linearly with basis set size.
- ACE performs well in the low-data limit, requiring orders of magnitude less data than the neural net models
- There is some optimisation to be had in selecting the optimal basis functions to include at each body order.
- One of the key advantages over GNN approaches is the data requirement.  MSE/training set size is a key performance descriptor.


Related Work
-----
__Chodera et al - arXiv:2010.01196v2__
An update to the OpenMM Sage forcefield that applies a graph convolutional neural network to learn continuous atomtypes using message passing, and a subsequent FFNN to generate body-ordered parameters compatible with empirical forcefield functional forms, giving performance on par with classical forcefields but it's much more easily extensible.

GNN approaches are not generally body-ordered, so the FFNN is required to convert the atomic representation to a set of body-ordered functions for evaluation by a classical MD engine

__General Notes__
- The authors show that by simply adding additional QM data to their model, they can produce parameters for a wide range of structures, achieving accuracy on par with classical forcefields.  This includes proteins and small molecules by simultaneously training on a Zinc subset and PepConf
- Their benchmark for energy comparison is not QML models explicitly, based on their relative speed.
- It is possible to simply transplant long-range interactions using VdW/Coulombic potentials straight from the classical potential, and use the QML/GNN potential to represent only the bonded interactions
- In principle, the same GPU optimisation employed here is transferrable to an ACE environment, whereas the ASE simulation engine is CPU-bound 
- In principle, the inherent body-ordering of terms in the ACE framework should cut out the 'middle man' neural net and can be used directly to evaluate the energy of the configuration.  The functional form is now polynomial instead of some fixed function
	- To what extent can GPU acceleration ameliorate the performance loss here? 



__PhysNet -  J. Chem. Theory Comput. 2019, 15, 3678−3693__
One of the forcefields compared in the ACE preprint is PhysNet, a message passing approach that does without physics-based descriptors and constructs them in a data-driven way, prior to feed forward layers to create the regression problem and predict molecular energies.

Nodes are featurised by basic cheminformatics descriptors, ring membership, formal charge etc.




Questions
-----
- To apply this to molecular modelling, a few requirements need to be met
	- Speed 
		- Need access to nanosecond timescales
		- Sparsify the set of basis functions
		- Implement GPU kernel
		- Combine with empirical forcefield? 
			- Or combine multiple models for different substrates (cheaper model for the peptide)
	- Accuracy - DFT-level energies would have a huge impact on FEP studies
	- Extensibility 
		- How general can the forcefields be?
		- How large a system can you simulate
- How would you treat long-range interactions? Charge Transfer events? How far from the equilibrium-ish simulated trajectory can you get?
	- If we want reactive potentials, we must be able to treat the long range physics sufficiently well.
	- Subtract this component before fitting if you know the form?
	- Apply a multi-scale model that combines the two length scales?
- How would you deal with solvation effects? Currently these are gas-phase calculations
- How do you go beyond systems that are accessable to high-level DFT
	- What do you do in the case of materials? Simulate on smaller systems?
	- Could you do the same for oligomers, polymers, etc?
- How general can the forcefields be made? Could we train a forcefield based on a congeneric series of ligands? 
- How would we model interactions with proteins? Could we calculate the change in binding energy using these potentials? 
- Can you apply explainability approahches (SHAP, ...) to identify most important contributions from basis sets?
- 
