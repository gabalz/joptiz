package ai.gandg.joptiz;

import ai.gandg.joptiz.DifferentiableFunction;


/** Various utility functions. */
public class Utils {

  /**
   * Estimates the gradient by forward or backward finite differentiation.
   *
   * @param fun the differentiable function
   * @param x the point around which the gradient is estimated
   * @param h the spacing parameter (positive for forward, negative for backward finite difference)
   * @param gradient placeholder of the gradient (or <code>null</code> to be allocated)
   * @return the estimated gradient
   */
  public static double[] finiteDifferenceGradient(DifferentiableFunction fun,
                                                  double[] x,
                                                  double h,
                                                  double[] gradient) {
    final int n = x.length;
    if (gradient == null) { gradient = new double[n]; }
    final double fx = fun.eval(x);
    for (int i = 0; i < n; ++i) {
      final double xi = x[i];
      x[i] += h;
      gradient[i] = (fun.eval(x) - fx) / h;
      x[i] = xi;
    }
    return gradient;
  }

  /**
   * Estimates the gradient by central finite differentiation.
   *
   * @param fun the differentiable function
   * @param x the point around which the gradient is estimated
   * @param h the spacing parameter (should be positive)
   * @param gradient placeholder of the gradient (or <code>null</code> to be allocated)
   * @return the estimated gradient
   */
  public static double[] centralFiniteDifferenceGradient(DifferentiableFunction fun,
                                                         double[] x,
                                                         double h,
                                                         double[] gradient) {
    assert h > 0.0;
    final int n = x.length;
    if (gradient == null) { gradient = new double[n]; }
    for (int i = 0; i < n; ++i) {
      final double xi = x[i];
      x[i] += 0.5 * h;
      final double fv = fun.eval(x);
      x[i] -= h;
      gradient[i] = (fv - fun.eval(x)) / h;
      x[i] = xi;
    }
    return gradient;
  }
}
