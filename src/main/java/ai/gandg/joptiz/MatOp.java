package ai.gandg.joptiz;


/** Matrix-vector operations used for the implementation of the optimization algorithms. */
final class MatOp {

  /** Copying a vector into another. */
  static void copy(double[] v, double[] out) {
    final int n = out.length;
    assert n == v.length;
    System.arraycopy(v, 0, out, 0, n);
  }

  /** Negating a vector. */
  static void neg(double[] v, double[] out) {
    final int n = out.length;
    assert n == v.length;
    for (int i = 0; i < n; ++i) { out[i] = -v[i]; }
  }

  /** Adding two vectors together. */
  static void add(double[] v1, double[] v2, double[] out) {
    final int n = out.length;
    assert n == v1.length;
    assert n == v2.length;
    for (int i = 0; i < n; ++i) { out[i] = v1[i] + v2[i]; }
  }

  /** Adding a scaled vector to another. */
  static void addScaled(double[] v1, double c, double[] v2, double[] out) {
    final int n = out.length;
    assert n == v1.length;
    assert n == v2.length;
    for (int i = 0; i < n; ++i) { out[i] = v1[i] + c * v2[i]; }
  }

  /** Subtracting a vector from another. */
  static void sub(double[] v1, double[] v2, double[] out) {
    final int n = out.length;
    assert n == v1.length;
    assert n == v2.length;
    for (int i = 0; i < n; ++i) { out[i] = v1[i] - v2[i]; }
  }

  /** Subtract a scaled vector from another. */
  static void subScaled(double[] v1, double c, double[] v2, double[] out) {
    final int n = out.length;
    assert n == v1.length;
    assert n == v2.length;
    for (int i = 0; i < n; ++i) { out[i] = v1[i] - c * v2[i]; }
  }

  /** Elementwise multiplying a vector by a constant. */
  static void mul(double[] v, double c, double[] out) {
    final int n = out.length;
    assert n == v.length;
    for (int i = 0; i < n; ++i) { out[i] = v[i] * c; }
  }

  /** Elementwise multiplying two vectors together. */
  static void mul(double[] v1, double[] v2, double[] out) {
    final int n = out.length;
    assert n == v1.length;
    assert n == v2.length;
    for (int i = 0; i < n; ++i) { out[i] = v1[i] * v2[i]; }
  }

  /** Dot product of two vectors. */
  static double dot(double[] v1, double[] v2) {
    final int n = v1.length;
    assert n == v2.length;
    double dp = v1[0] * v2[0];
    for (int i = 1; i < n; ++i) { dp += v1[i] * v2[i]; }
    return dp;
  }

  /** Calculating the 2-norm of a vector. */
  static double norm2(double[] v) {
    final int n = v.length;
    double sq = 0.0;
    for (int i = 0; i < n; ++i) {
      final double vi = v[i];
      sq += vi * vi;
    }
    return Math.sqrt(sq);
  }

  /** Calculating the inf-norm of a vector. */
  static double normInf(double[] v) {
    final int n = v.length;
    double m = 0.0;
    for (int i = 0; i < n; ++i) {
      final double absvi = Math.abs(v[i]);
      if (absvi > m) { m = absvi; }
    }
    return m;
  }
}
