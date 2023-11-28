package ai.gandg.joptiz;

import ai.gandg.joptiz.LineSearch;


public class BacktrackingLineSearch implements LineSearch {
  private double tau = 0.5;
  private double fvTol = 1e-4;
  private double minStepSize = 1e-16;
  private int nFunEval = 0;

  public BacktrackingLineSearch() {
  }

  public BacktrackingLineSearch(double tau, double funTol, double minStepSize) {
    this.tau = tau;
    this.fvTol = fvTol;
    this.minStepSize = minStepSize;
  }

  /**
   * Returns the shrinkage parameter which is used to scale the step size in each iterations.
   *
   * @return the shrinkage parameter
   */
  public double getShrinkageParameter() {
    return tau;
  }

  /**
   * Sets the shrinkage parameter which is used to scale the step size in each iterations.
   *
   * @param tau the new value of the shrinkage parameter
   */
  public void setShrinkageParameter(double tau) {
    this.tau = tau;
  }

  /**
   * Returns the function value tolerance used for the Armijo-Goldstein condition.
   *
   * @return the function value tolerance
   */
  public double getFunctionValueTolerance() {
    return fvTol;
  }

  /**
   * Sets the function value tolerance used for the Armijo-Goldstein condition.
   *
   * @param fvTol the new value of the function value tolerance
   */
  public void setFunctionValueTolerance(double fvTol) {
    this.fvTol = fvTol;
  }

  @Override
  public double getMinimumStepSize() {
    return minStepSize;
  }

  /**
   * Sets the minimum step size.
   * 
   * @param minstepsize the new value of the minimum step size
   */
  public void setMinimumStepSize(double minStepSize) {
    this.minStepSize = minStepSize;
  }

  @Override
  public int getNumberOfFunctionEvaluations() {
    return nFunEval;
  }

  @Override
  public double minimize(DifferentiableFunction fun, double[] x0,
                         double f0, double[] g0, double alpha0,
                         double[] p, double[] x) {
    final double fvTolM = fvTol * MatOp.dot(g0, p);
    double alpha = 1.0;
    MatOp.add(x0, p, x); // x = x0 + p
    nFunEval = 0;
    while (alpha >= minStepSize) {
      final double f = fun.eval(x);
      ++nFunEval;
      if (f - f0 <= alpha * fvTolM) { break; }
      alpha *= tau;
      MatOp.addScaled(x0, alpha, p, x); // x = x0 + alpha * p
    }
    if (alpha < minStepSize) {
      alpha = -1.0;
    }
    return alpha;
  }
}
