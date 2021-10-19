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

$$ A_{z_i znlm} = \langle \rho_i^{z} | \phi_{nlm}^{z_i z} \rangle = \sum_j \phi_{nlm}^{z_i z_j} (r_{ij})$$

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

