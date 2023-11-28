package ai.gandg.joptiz.test;

import org.junit.Test;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertArrayEquals;

import java.util.Arrays;

import ai.gandg.joptiz.DifferentiableFunction;
import ai.gandg.joptiz.LBFGS;
import ai.gandg.joptiz.LBFGS.Status;


public class LBFGSTest {
  final double TOL = 1e-8;

  @Test public void readmeExampleTest() {
    class QuadFun implements DifferentiableFunction {
      @Override
      public double eval(double[] x) {
        return 0.5 * (x[0]*x[0] + x[1]*x[1]);
      }
      
      @Override
      public double eval(double[] x, double[] gradient) {
        System.arraycopy(x, 0, gradient, 0, x.length);
        return eval(x);
      }
    }

    LBFGS lbfgs = new LBFGS(2, 20);
    double fOpt = lbfgs.minimize(new QuadFun(), new double[]{-1.0, 2.0});
    double[] xOpt = lbfgs.getSolutionVector();
    
    assertEquals(0.0, fOpt, TOL);
    assertArrayEquals(new double[]{0.0, 0.0}, xOpt, TOL);
  }

  @Test public void smallQuad1Test() {
    class QuadFun implements DifferentiableFunction {
      final double x0 = 1.0;
      final double x1 = -2.0;

      @Override
      public double eval(double[] x) {
        final double diff0 = x[0] - x0;
        final double diff1 = x[1] - x1;
        return 0.5 * (diff0*diff0 + diff1*diff1);
      }

      @Override
      public double eval(double[] x, double[] gradient) {
        final double diff0 = x[0] - x0;
        final double diff1 = x[1] - x1;
        gradient[0] = diff0;
        gradient[1] = diff1;
        return 0.5 * (diff0*diff0 + diff1*diff1);
      }
    }
    DifferentiableFunction fun = new QuadFun();
    double[] x0 = new double[2];

    double f = 0.0;
    double[] x = null;

    LBFGS lbfgs = new LBFGS(5, 20);
    x0[0] = 5.0; x0[1] = 3.0;
    assertEquals(20.5, fun.eval(x0), TOL);
    f = lbfgs.minimize(fun, x0);
    x = lbfgs.getSolutionVector();
    // System.out.println("smallQuad1, f: " + f + ", x: [" + x[0] + ", " + x[1] + "]");
    assertEquals(0.0, f, TOL);
    assertEquals(Status.SUCCESS, lbfgs.getStatus());
    assertEquals(f, lbfgs.getSolutionValue(), TOL);
    assertArrayEquals(new double[]{1.0, -2.0}, x, TOL);
    assertEquals(f, fun.eval(x), TOL);
    assertEquals(2, lbfgs.getNumberOfIterations());
    assertEquals(3, lbfgs.getNumberOfFunctionEvaluations());
  }

  @Test public void smallQuad2Test() {
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

      @Override
      public double eval(double[] x, double[] gradient) {
        final double diff0 = x[0] - x0;
        final double diff1 = x[1] - x1;
        final double sum = x[0] + 2.0*x[1];
        gradient[0] = diff0 + sum;
        gradient[1] = diff1 + 2.0*sum;
        return 0.5 * (diff0*diff0 + diff1*diff1 + sum*sum);
      }
    }
    DifferentiableFunction fun = new QuadFun();
    double[] x0 = new double[2];

    double f = 0.0;
    double[] x = null;

    LBFGS lbfgs = new LBFGS(5, 25);
    x0[0] = 5.0; x0[1] = 3.0;
    assertEquals(81.0, fun.eval(x0), TOL);
    f = lbfgs.minimize(fun, x0);
    x = lbfgs.getSolutionVector();
    // System.out.println("smallQuad2, f: " + f + ", x: [" + x[0] + ", " + x[1] + "]");
    assertEquals(0.75, f, TOL);
    assertEquals(Status.SUCCESS, lbfgs.getStatus());
    assertEquals(f, lbfgs.getSolutionValue(), TOL);
    assertArrayEquals(new double[]{1.500000447,
                                   -0.99999949}, x, TOL);
    assertEquals(f, fun.eval(x), TOL);
    assertEquals(13, lbfgs.getNumberOfIterations());
    assertEquals(74, lbfgs.getNumberOfFunctionEvaluations());
  }

  @Test public void himmelblauTest() { // Himmelblau function
    class HBFun implements DifferentiableFunction {

      @Override
      public double eval(double[] x) {
        final double v1 = x[0]*x[0] + x[1] - 11.0;
        final double v2 = x[0] + x[1]*x[1] - 7.0;
        return v1*v1 + v2*v2;
      }

      @Override
      public double eval(double[] x, double[] gradient) {
        final double v1 = x[0]*x[0] + x[1] - 11.0;
        final double v2 = x[0] + x[1]*x[1] - 7.0;
        gradient[0] = 4.0 * v1 * x[0] + 2.0 * v2;
        gradient[1] = 2.0 * v1 + 4.0 * v2 * x[1];
        return v1*v1 + v2*v2;
      }
    }
    DifferentiableFunction fun = new HBFun();
    double[] x0 = new double[2];
    
    double f = 0.0;
    double[] x = null;

    LBFGS lbfgs = new LBFGS(2, 50);
    x0[0] = 1.0; x0[1] = 1.0;
    assertEquals(106.0, fun.eval(x0), TOL);
    f = lbfgs.minimize(fun, x0);
    x = lbfgs.getSolutionVector();
    // System.out.println("Himmelblau, f: " + f + ", x: [" + x[0] + ", " + x[1] + "]");
    assertEquals(0.0, f, TOL);
    assertEquals(Status.SUCCESS, lbfgs.getStatus());
    assertEquals(6, lbfgs.getNumberOfRestarts());
    assertEquals(25, lbfgs.getNumberOfIterations());
    assertEquals(129, lbfgs.getNumberOfFunctionEvaluations());
    assertEquals(f, lbfgs.getSolutionValue(), TOL);
    assertArrayEquals(new double[]{2.999999839,
                                   1.999999694}, x, TOL);
  }

  @Test public void boothTest() { // Booth function
    class BoothFun implements DifferentiableFunction {

      @Override
      public double eval(double[] x) {
        final double v1 = x[0] + 2.0*x[1] - 7.0;
        final double v2 = 2.0*x[0] + x[1] - 5.0;
        return v1*v1 + v2*v2;
      }

      @Override
      public double eval(double[] x, double[] gradient) {
        final double v1 = x[0] + 2.0*x[1] - 7.0;
        final double v2 = 2.0*x[0] + x[1] - 5.0;
        gradient[0] = 2.0 * v1 + 4.0 * v2;
        gradient[1] = 4.0 * v1 + 2.0 * v2;
        return v1*v1 + v2*v2;
      }
    }
    DifferentiableFunction fun = new BoothFun();
    double[] x0 = new double[2];
    
    double f = 0.0;
    double[] x = null;

    LBFGS lbfgs = new LBFGS(2, 50);
    x0[0] = 0.0; x0[1] = 0.0;
    assertEquals(74.0, fun.eval(x0), TOL);
    f = lbfgs.minimize(fun, x0);
    x = lbfgs.getSolutionVector();
    // System.out.println("Booth, f: " + f + ", x: [" + x[0] + ", " + x[1] + "]");
    assertEquals(0.0, f, TOL);
    assertEquals(Status.SUCCESS, lbfgs.getStatus());
    assertEquals(7, lbfgs.getNumberOfRestarts());
    assertEquals(39, lbfgs.getNumberOfIterations());
    assertEquals(226, lbfgs.getNumberOfFunctionEvaluations());
    assertEquals(f, lbfgs.getSolutionValue(), TOL);
    assertArrayEquals(new double[]{0.999998201,
                                   3.000002647}, x, TOL);
  }

  @Test public void sumsqTest() { // Sum-Squares function
    class SumSquaresFun implements DifferentiableFunction {

      @Override
      public double eval(double[] x) {
        double v = 0.0;
        for (int i = 0; i < x.length; ++i) {
          final double xi = x[i];
          v += (i+1) * xi*xi;
        }
        return v;
      }

      @Override
      public double eval(double[] x, double[] gradient) {
        double v = 0.0;
        for (int i = 0; i < x.length; ++i) {
          final double xi = x[i];
          final int i1 = i + 1;
          v += i1 * xi*xi;
          gradient[i] = (2*i1) * xi;
        }
        return v;
      }
    }
    final int n = 20;
    DifferentiableFunction fun = new SumSquaresFun();
    double[] x0 = new double[n];
    
    double f = 1.0;
    double[] x = null;

    LBFGS lbfgs = new LBFGS(5, 150);
    Arrays.fill(x0, 1.0);
    f = lbfgs.minimize(fun, x0);
    x = lbfgs.getSolutionVector();
    // System.out.println("SumSquares(n:" + n + "), f: " + f);
    assertEquals(0.0, f, TOL);
    assertEquals(f, lbfgs.getSolutionValue(), TOL);
    assertEquals(Status.SUCCESS, lbfgs.getStatus());
    assertEquals(22, lbfgs.getNumberOfRestarts());
    assertEquals(92, lbfgs.getNumberOfIterations());
    assertEquals(492, lbfgs.getNumberOfFunctionEvaluations());
  }

  @Test public void tridTest() { // Trid function
    class TridFun implements DifferentiableFunction {

      @Override
      public double eval(double[] x) {
        double v = 0.0;
        double prev = 0.0;
        for (int i = 0; i < x.length; ++i) {
          final double xi = x[i];
          final double xi1 = xi - 1.0;
          v += xi1 * xi1 - prev * xi;
          prev = xi;
        }
        return v;
      }

      @Override
      public double eval(double[] x, double[] gradient) {
        final int n = x.length;
        double v = 0.0;
        double prev = 0.0;
        for (int i = 0; i < n; ++i) {
          final double xi = x[i];
          final double xi1 = xi - 1.0;
          v += xi1 * xi1 - prev * xi;
          gradient[i] = 2.0 * xi1 - prev;
          if (i+1 < n) { gradient[i] -= x[i+1]; }
          prev = xi;
        }
        return v;
      }
    }
    final int n = 10;
    DifferentiableFunction fun = new TridFun();
    double[] x0 = new double[n];

    final double fOpt = -n*(n+4)*(n-1) / 6.0;
    final double[] xOpt = new double[n];
    for (int i = 0; i < n; ++i) { xOpt[i] = (i+1)*(n-i); }
    final double TOL = 1e-8 * n;

    double f = 1.0;
    double[] x = null;

    LBFGS lbfgs = new LBFGS(5, 100);
    Arrays.fill(x0, 0.0);
    f = lbfgs.minimize(fun, x0);
    x = lbfgs.getSolutionVector();
    // System.out.println("Trid(n:" + n + "), f: " + f);
    assertEquals(fOpt, f, TOL);
    assertEquals(f, lbfgs.getSolutionValue(), TOL);
    assertEquals(Status.SUCCESS, lbfgs.getStatus());
    assertEquals(9, lbfgs.getNumberOfRestarts());
    assertEquals(59, lbfgs.getNumberOfIterations());
    assertEquals(498, lbfgs.getNumberOfFunctionEvaluations());
    assertArrayEquals(xOpt, lbfgs.getSolutionVector(), 1e-3);
  }
}
