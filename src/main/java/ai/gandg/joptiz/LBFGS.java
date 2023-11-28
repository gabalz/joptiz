package ai.gandg.joptiz;

import ai.gandg.joptiz.BacktrackingLineSearch;
import ai.gandg.joptiz.LineSearch;
import ai.gandg.joptiz.MatOp;


/**
 * The Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) algorithm.
 */
public class LBFGS {
  private int maxMemorySize;
  private int maxIter;
  private LineSearch ls;
  private double accTol = 1e-5;
  private double curveTol = 1e-8;

  private int nIter = 0;
  private int nFunEval = 0;
  private int nRestarts = 0;
  private Status status = Status.SUCCESS;

  private double f;
  private double[] x, xNew, p, g, gOld, alphas, rhos;
  private double[][] sMat, yMat;

  /**
   * Possible reasons of termination.
   */
  public static enum Status {

    /** The gradient <code>norm2</code> became smaller
     *  than <code>getAccuracyTolerance() * max(1,norm2(x))</code>,
     *  where <code>x</code> denotes the solution vector.
     */
    SUCCESS,

    /** The initial point <code>x0</code> is the minimizer. */
    ALREADY_MINIMIZED,

    /** The line search could not decrease the function value. */
    TOO_SMALL_STEPSIZE,

    /** The maximum number of iterations has been reached. */
    MAX_ITER_REACHED,
  }

  /**
   * Constructor of a L-BFGS optimization object.
   *
   * @param maxMemorysize maximum number of gradients to be stored
   * @param maxIter maximum number of iterations
   * @param ls the line search algorithm
   */
  public LBFGS(int maxMemorySize, int maxIter, LineSearch ls) {
    this.maxMemorySize = maxMemorySize;
    this.maxIter = maxIter;
    this.ls = ls;
  }

  /**
   * Constructor of a L-BFGS optimization object.
   *
   * @param maxMemorysize maximum number of gradients to be stored
   * @param maxIter maximum number of iterations
   */
  public LBFGS(int maxMemorySize, int maxIter) {
    this.maxMemorySize = maxMemorySize;
    this.maxIter = maxIter;
    this.ls = new BacktrackingLineSearch();
  }

  /**
   * Returns the maximum number of gradient and position vector deltas to be stored.
   *
   * @return the maximum number of gradient and position vector deltas to be stored
   */
  public int getMaximumMemorySize() {
    return maxMemorySize;
  }

  /**
   * Sets the maximum number of gradient and position vector deltas to be stored.
   *
   * @param maxMemorySize the new value of maximum memory size
   */
  public void setMaximumMemorySize(int maxMemorySize) {
    this.maxMemorySize = maxMemorySize;
  }

  /**
   * Returns the maximum number of iterations.
   *
   * @return the maximum number of iterations
   */
  public int getMaximumNumberOfIterations() {
    return maxIter;
  }

  /**
   * Sets the maximum number of iterations.
   *
   * @param maxIter the new value of maximum number of iterations
   */
  public void setMaximumNumberOfIterations(int maxIter) {
    this.maxIter = maxIter;
  }

  /**
   * Returns the line search method.
   *
   * @return the line search method
   */
  public LineSearch getLineSearch() {
    return ls;
  }

  /**
   * Sets the line search method.
   */
  public void setLineSearch(LineSearch ls) {
    this.ls = ls;
  }

  /**
   * Returns the accuracy tolerance parameter.
   * The minimization terminates when <code>norm2(g) < accuracyTolerance * max(1, norm2(x))</code>.
   *
   * @return the accuracy tolerance parameter
   */
  public double getAccuracyTolerance() {
    return accTol;
  }

  /**
   * Sets the accuracy tolerance parameter.
   * The minimization terminates when <code>norm2(g) < accuracyTolerance * max(1, norm2(x))</code>.
   *
   * @param the new value of the accuracy tolerance parameter
   */
  public void setAccuracyTolerance(double accTol) {
    this.accTol = accTol;
  }

  private void allocateWorkspace(int n) {
    final boolean isCompleteReallocNeeded = (x == null || x.length != n);
    if (isCompleteReallocNeeded) {
      x = new double[n];
      xNew = new double[n];
      p = new double[n];
      g = new double[n];
      gOld = new double[n];
    }
    if (isCompleteReallocNeeded || sMat == null || sMat.length != maxMemorySize) {
      sMat = new double[maxMemorySize][n];
      yMat = new double[maxMemorySize][n];
      alphas = new double[maxMemorySize];
      rhos = new double[maxMemorySize];
    }
  }

  /**
   * Returns the solution value of the last minimization.
   *
   * @return the solution value of the last minimization
   */
  public double getSolutionValue() {
    return f;
  }

  /**
   * Returns the solution vector of the last minimization.
   *
   * @return the solution vector of the last minimization
   */
  public double[] getSolutionVector() {
    return x;
  }

  /**
   * Returns the number of iterations of the last minimization.
   *
   * @return the number of iterations of the last minimization
   */
  public int getNumberOfIterations() {
    return nIter;
  }

  /**
   * Returns the number of function evaluations of the last minimization.
   *
   * @return the number of function evaluations of the last minimization
   */
  public int getNumberOfFunctionEvaluations() {
    return nFunEval;
  }

  /**
   * Returns the number of restarts for the estimation of the Hessian
   * due to too small curvature during the last minimization.
   *
   * @return the number of restarts for the Hessian estimation
   */
  public int getNumberOfRestarts() {
    return nRestarts;
  }

  /**
   * Returns the status (reason of termination) of the last minimization.
   *
   * @return the status (reason of termination) of the last minimization
   */
  public Status getStatus() {
    return status;
  }

  /**
   * Minimizing a differentiable function.
   *
   * @param fun the function to be minimized
   * @param x0 the starting point
   * @return objective value at the minimum
   */
  public double minimize(DifferentiableFunction fun, double[] x0) {
    final int n = x0.length;
    allocateWorkspace(n);

    MatOp.copy(x0, x); // x = x0
    int memPos = 0, memSize = 0;
    double gamma = 1.0;
    f = fun.eval(x, g); // evaluate the initial point
    nFunEval = 1;

    double gnorm = MatOp.norm2(g);
    double xnorm = MatOp.norm2(x);
    if (gnorm < accTol * Math.max(1.0, xnorm)) {
      status = Status.ALREADY_MINIMIZED;
      return f;
    }
    double alpha = 1.0 / Math.max(gnorm, ls.getMinimumStepSize());

    nIter = 0;
    nRestarts = 0;
    while (true) {
      if (nIter == maxIter) {
        status = Status.MAX_ITER_REACHED;
        break;
      }
      ++nIter;

      if (gnorm < accTol * Math.max(1.0, xnorm)) {
        status = Status.SUCCESS;
        break;
      }

      lbfgsNegSearchDir(memPos, memSize, gamma); // computes p
      alpha = ls.minimize(fun, x, f, g, alpha, p, xNew);
      nFunEval += ls.getNumberOfFunctionEvaluations();
      if (alpha < 0.0) {
        status = Status.TOO_SMALL_STEPSIZE;
        break;
      }

      final double fOld = f;
      // Swapping gOld and g:
      double[] tmp = gOld;
      gOld = g;
      g = tmp;

      f = fun.eval(xNew, g);
      ++nFunEval;

      if (++memPos == maxMemorySize) { memPos = 0; }
      if (memSize <= maxMemorySize) { ++memSize; }
      final double[] s = sMat[memPos];
      final double[] y = yMat[memPos];

      MatOp.sub(xNew, x, s); // s = xNew - x;
      MatOp.sub(g, gOld, y); // y = g - gOld
      final double sTy = MatOp.dot(s, y);
      if (Math.abs(sTy) < curveTol) {
        ++nRestarts;
        memPos = memSize = 0;
        gamma = 1.0;
      }
      else {
        rhos[memPos] = 1.0 / sTy;
        gamma = sTy / MatOp.dot(y, y);
      }

      // Swapping x and xNew:
      tmp = x;
      x = xNew;
      xNew = tmp;

      gnorm = MatOp.norm2(g);
      xnorm = MatOp.norm2(x);
      alpha = 1.0;
    }

    return f;
  }

  private void lbfgsNegSearchDir(int memPos, int memSize, double gamma) {
    MatOp.neg(g, p); // p = -g
    int i = memPos;
    for (int c = 0; c < memSize; ++c) {
      if (--i < 0) { i = maxMemorySize - 1; }
      alphas[i] = rhos[i] * MatOp.dot(sMat[i], p);
      MatOp.subScaled(p, alphas[i], yMat[i], p); // p -= alpha[i] * yMat[i]
    }
    MatOp.mul(p, gamma, p); // p *= gamma
    for (int c = 0; c < memSize; ++c) {
      final double beta = rhos[i] * MatOp.dot(yMat[i], p);
      MatOp.addScaled(p, alphas[i] - beta, sMat[i], p); // p += sMat[i] * (alphas[i] - beta)
    }
  }
}
