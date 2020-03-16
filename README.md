# Bayesian or Probabilistic Graphical Models in tensorflow

Probabilistic graphical models are the models where, you can define your problem statement in the form of graphs.

Say, x -> y is a graph.

The above statement can be interpreted as, you are given a set of values x, which influence y.

Typically, x and y can be any scalar variables, but mostly it would help us if we consider x and y as distributions

Given a set of values of x, we need to find y. This problem can be defined as a kind of linear regression problem, 
with x as features and y as observation.
The values observed for y are supposed to be Expected value of distribution of y.
Since Y is influenced by x, if we are given a value of x, it can influence
the distribution of y. So, if we are given a value of x, to predict y, we need to find
conditional distribution of y given x.

So, now we have, x - independent variables, y - Expected value of distribution
given x

We can build the objective function as Maximum likelihood function, given the 
assumption of distribution of y. 

If y is a continuous variable, we can assume P(y|x) to have a Gaussian distribution.
If y is a multinomial variable, we can assume P(y|x) to have a Multinomial distribution.

In this code, I have written the objective function in tensorflow, which will
calculate the Gradients by itself.

There is one file for Gaussian Distribution and one file for Multinomial Distribution.