# AutoDyn

A Python library for Lie-controllability anlyses on dynamical systems models.
Made with whole-brain models of brain activity and DBS in mind.


## What is Lie Controllability
Lie controllability (LC) is the nonlinear version of controllability analyses.
In LC we can directly observe the interactions that our controller has on the dynamics of the system, and then use these interactions to design controllers with desired properties.

## Why this library?
At the time I started this there weren't any available libraries that would be able to interact with the whole-brain models I fit to empirical data (DBS for Depression work).
While working with the group at The Virtual Brain (TVB) I found a need to assess DBS ability to control behavior through the brain state/dynamics.
This work was an effort to build a library, from scratch.

Since starting this there has been great progress in nonlinear controllability analyses, but the hope is that this library is an easy-to-use drop in for toy-models of the brain.
Many of the models that clinicians use in their day-to-day job can easily be collapsed into simple toy-models and this library is meant to analyse those toy-models.
Other libraries would be more appropriate for efforts to model massive whole-brain networks derived from massive datasets.
