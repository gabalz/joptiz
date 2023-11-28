package ai.gandg.joptiz.test;

import org.junit.Test;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertArrayEquals;

import ai.gandg.joptiz.DifferentiableFunction;
import static ai.gandg.joptiz.Utils.*;


public class UtilsTest {
  final double TOL = 1e-8;

  @Test public void finiteDiffGradientTest() {
    class QuadFun implements DifferentiableFunction {
      final double x0 = 1.0;
      final double x1 = -2.0;

      @Override
      public double eval(double[] x) {
        final double diff0 = x[0] - x0;
        final double diff1 = x[1] - x1;
        final double sum = x[0] + 2.0*x[1];
        return 0.5 * (diff0*diff0 + diff1*diff1 + sum*sum);
      }
      public double eval(double[] x, double[] gradient) {
        final double diff0 = x[0] - x0;
        final double diff1 = x[1] - x1;
        final double sum = x[0] + 2.0*x[1];
        gradient[0] = diff0 + sum;
        gradient[1] = diff1 + 2.0*sum;
        return 0.5 * (diff0*diff0 + diff1*diff1 + sum*sum);
      }
    }
    QuadFun fun = new QuadFun();

    double[] x = new double[]{5.0, 3.0};
    double[] g = new double[2];
    double[] gFD = new double[2];

    double f = fun.eval(x, g);
    assertEquals(81.0, f, TOL);
    assertEquals(f, fun.eval(x), TOL);
    assertArrayEquals(new double[]{15.0, 27.0}, g, TOL);

    // forward finite differentiation
    finiteDifferenceGradient(fun, x, 1e-6, gFD);
    assertArrayEquals(new double[]{15.000001014,
                                   27.000002504}, gFD, TOL);    

    // backward finite differentiation
    finiteDifferenceGradient(fun, x, -1e-6, gFD);
    assertArrayEquals(new double[]{14.99999901,
                                   26.999997502}, gFD, TOL);    

    // central finite differentiation
    centralFiniteDifferenceGradient(fun, x, 1e-6, gFD);
    assertArrayEquals(new double[]{14.999999991,
                                   27.000000046}, gFD, TOL);    
  }
}
