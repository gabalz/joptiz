package ai.gandg.joptiz.test;

import org.junit.Test;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertArrayEquals;

import ai.gandg.joptiz.BacktrackingLineSearch;
import ai.gandg.joptiz.DifferentiableFunction;
import ai.gandg.joptiz.LineSearch;


public class BacktrackingLineSearchTest {
  final double TOL = 1e-8;

  @Test public void quadFunTest() {
    class QuadFun implements DifferentiableFunction {
      final double x0 = 1.0;
      final double x1 = -2.0;

      @Override
      public double eval(double[] x) {
        final double diff0 = x[0] - x0;
        final double diff1 = x[1] - x1;
        return 0.5 * (diff0*diff0 + diff1*diff1);
      }
      public double eval(double[] x, double[] gradient) {
        final double diff0 = x[0] - x0;
        final double diff1 = x[1] - x1;
        gradient[0] = diff0;
        gradient[1] = diff1;
        return 0.5 * (diff0*diff0 + diff1*diff1);
      }
    }
    QuadFun fun = new QuadFun();
    double f0;
    double[] x0 = new double[2];
    double[] g0 = new double[2];
    double[] x = new double[2]; 
    double[] p = new double[2];
    LineSearch ls = new BacktrackingLineSearch();

    x0[0] = 5.0; x0[1] = 1.0;
    p[0] = -1.0; p[1] = 0.0;
    f0 = fun.eval(x0, g0);
    assertEquals(1.0, ls.minimize(fun, x0, f0, g0, 1.0, p, x), TOL);
    assertArrayEquals(new double[]{4.0, 1.0}, x, TOL);

    x0[0] = 5.0; x0[1] = 1.0;
    p[0] = 1.0; p[1] = 0.0; // ascent direction
    f0 = fun.eval(x0, g0);
    assertEquals(0.0, ls.minimize(fun, x0, f0, g0, 1.0, p, x), 1e-15);
    assertArrayEquals(x0, x, ls.getMinimumStepSize());

    x0[0] = 5.0; x0[1] = 1.0;
    f0 = fun.eval(x0, g0);
    p[0] = -g0[0]; p[1] = -g0[1];
    assertEquals(1.0, ls.minimize(fun, x0, f0, g0, 1.0, p, x), TOL);
    assertArrayEquals(new double[]{1.0, -2.0}, x, TOL);
    assertEquals(0.0, fun.eval(x), TOL);

    x0[0] = 5.0; x0[1] = 1.0;
    f0 = fun.eval(x0, g0);
    p[0] = -5*g0[0]; p[1] = -4*g0[1];
    assertEquals(0.25, ls.minimize(fun, x0, f0, g0, 1.0, p, x), TOL);
    assertArrayEquals(new double[]{0.0, -2.0}, x, TOL);
    assertEquals(0.5, fun.eval(x), TOL);
  }
}
