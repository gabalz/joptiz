[![Build](https://github.com/gabalz/joptiz/actions/workflows/maven.yml/badge.svg)](https://github.com/gabalz/joptiz/actions/workflows/maven.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# JOptiZ

JOptiZ is an optimization library implemented in Java providing garbage collection free operations. Its aim to rely on minimal dependencies and stay compilable on embedded Java implementations.

Supported optimization algorithms:

  - Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) algorithm.

## USAGE EXAMPLES

  ```
  import ai.gandg.joptiz.DifferentiableFunction;
  import ai.gandg.joptiz.LBFGS;
  ```

  Minimizing a simple quadratic fuinction:

  ```
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
  ```

## COMPILATION

  After manually downloading the source code, it can be compiled by: 

  ```
  mvn package
  ```

  This will create the `joptiz-<version>.jar` file under the `target` directory.

## DOCUMENTATION

  ```
  mvn javadoc:javadoc
  ```

  This will generate the API documentation into `target/site/apidocs`.

## TESTS AND BENCHMARKS

### Running tests

  Running all test cases:

  ```
  mvn test
  ```

  Running a single test case (e.g., `himmelblauTest` in `LBFGSTests`):

  ```
  mvn -Dtest=LBFGSTests#himmelblauTest test
  ```


