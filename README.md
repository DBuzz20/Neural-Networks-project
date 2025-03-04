# Machine Learning Project - Neural Networks
Project for the "Optimization Methods for Machine Learning" course.

Master's degree course in "Business Intelligence and Analytics" (Management engineering's curricula).

## Project Description
The project, developed in `Python`, involves the implementation of various approaches and techniques regarding `Neural Networks` in machine learning, putting particular emphasis on the optimization side of the developement.
It consisted in constructing functioning neural networks from skratch, without using any existing library for their creation and management.

In particular, the project required knowing and coding concepts such as: FeedForward NN (with full MLP and full RBF), gradient method (implemented manually through a gradient function), grid search with kfold cross validation, extreme MLP, unsupervised RBF and two-block decomposition method (alternating convex and non-convex optimization).

All the work conducted had the aim of optimization. Infact, each question has been solved using some optimization routine involving `scipy.optimize.minimize` and/or `np.linalg.lstsq`, depending on the nature of the optimization problem in analysis.

## Files
The `Assignment` folder contains all the details about the requests of each question, togheter with an image of the expected plot that my code should produce after the training.

The `Code` folder presents the actual code, divided by question. In it, are also available the plots produced by each question's implementation.

Also, the `Final_report.pdf` file describes the approach followed when developing such questions, explaining the decisions taken, with an eye on the over/under fitting possibilities, and all the performance results of such implementations (both in time and precision).
The report also provides various graphs and comments about the development, including comparisons between all the developed techiniques.
