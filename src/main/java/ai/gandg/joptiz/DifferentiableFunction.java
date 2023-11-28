package ai.gandg.joptiz;


/**
 * Interface specifying the operations of a differentiable function.
 */
public interface DifferentiableFunction {
  /**
   * Evaluation of the function value at position <code>x</code>.
   *
   * @param x position vector
   * @return function value at <code>x</code>
   */
  double eval(double[] x);

  /**
   * Evaluation of the function value and the gradient at position <code>x</code>.
   *
   * @param x position vector
   * @param gradient placeholder for the computation of the gradient at <code>x</code>
   * @return function value at <code>x</code>
   */
  double eval(double[] x, double[] gradient);
}
