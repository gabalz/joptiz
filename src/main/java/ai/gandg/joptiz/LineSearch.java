package ai.gandg.joptiz;

import ai.gandg.joptiz.DifferentiableFunction;


/**
 * Interface specifying the operations of a line search algorithm.
 */
public interface LineSearch {

  /**
   * Returns the minimum step size.
   *
   * @return the minimum step size
   */
  double getMinimumStepSize();

  /**
   * Returns the number of function evluations of the last minimization.
   */
  int getNumberOfFunctionEvaluations();

  /*
   * Minimize a differentiable function along a search direction.
   *
   * @param fun the function to be minimized
   * @param x0 the starting point
   * @param f0 the function value of at the starting point <code>x0</code>
   * @param g0 the gradient at the starting point <code>x0</code>
   * @param alpha0 the initial step size
   * @param p the search direction
   * @param x placeholder for the new point <code>x0 + alpha*p</code>
   * @return the step size <code>alpha</alpha>
   */
  double minimize(DifferentiableFunction fun, double[] x0,
                  double f0, double[] g0, double alpha0,
                  double[] p, double[] x);
}
