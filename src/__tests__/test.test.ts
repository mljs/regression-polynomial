import { NumberArray } from 'cheminfo-types';
import { createRandomArray, xSum } from 'ml-spectra-processing';
import { expect, it, describe } from 'vitest';

import { PolynomialRegression } from '..';

function assertCoefficientsAndPowers(
  result: PolynomialRegression,
  expectedCs: NumberArray,
  expectedPowers: NumberArray,
) {
  for (let i = 0; i < expectedCs.length; ++i) {
    expect(result.coefficients[i]).toBeCloseTo(expectedCs[i], 10e-6);
    expect(result.powers).toStrictEqual(expectedPowers);
  }
  expect(result.degree).toBe(Math.max(...expectedPowers));
}

describe('Polynomial regression', () => {
  it('basic linear test', () => {
    const size = 1000;
    const x = new Array(size).fill(0).map((_, i) => i);
    const y = new Array(size).fill(1);
    const regression = new PolynomialRegression(x, y, 1, {
      interceptAtZero: false,
    });
    let difference = 0;
    for (let i = 0; i < size; i++) {
      difference += Math.abs(regression.predict(x[i]) - y[i]);
    }
    expect(difference).closeTo(0, 1e-6);
  });
  it('degree 2', () => {
    const x = [-3, 0, 2, 4];
    const y = [3, 1, 1, 3];
    const result = new PolynomialRegression(x, y, 2);
    const expectedCoefficients = [0.850519, -0.192495, 0.178462];
    const expectedPowers = [0, 1, 2];

    assertCoefficientsAndPowers(result, expectedCoefficients, expectedPowers);

    const score = result.score(x, y);
    expect(score.r2).toBeGreaterThan(0.8);
    expect(score.chi2).toBeLessThan(0.1);
    expect(score.rmsd).toBeCloseTo(0.12);
    expect(result.toString(4)).toBe(
      'f(x) = 0.1785 * x^2 - 0.1925 * x + 0.8505',
    );
    expect(result.toLaTeX(2)).toBe('f(x) = 0.18x^{2} - 0.19x + 0.85');
  });

  it('degree 2 typed array', () => {
    const x = new Float64Array([-3, 0, 2, 4]);
    const y = new Float64Array([3, 1, 1, 3]);
    const result = new PolynomialRegression(x, y, 2);

    const expectedCoefficients = [0.850519, -0.192495, 0.178462];
    const expectedPowers = [0, 1, 2];

    assertCoefficientsAndPowers(result, expectedCoefficients, expectedPowers);

    const score = result.score(x, y);
    expect(score.r2).toBeGreaterThan(0.8);
    expect(score.chi2).toBeLessThan(0.1);
    expect(score.rmsd).toBeCloseTo(0.12);
    expect(result.toString(4)).toBe(
      'f(x) = 0.1785 * x^2 - 0.1925 * x + 0.8505',
    );
    expect(result.toLaTeX(2)).toBe('f(x) = 0.18x^{2} - 0.19x + 0.85');
  });

  it('degree 5', () => {
    const x = [50, 50, 50, 70, 70, 70, 80, 80, 80, 90, 90, 90, 100, 100, 100];
    const y = [
      3.3, 2.8, 2.9, 2.3, 2.6, 2.1, 2.5, 2.9, 2.4, 3.0, 3.1, 2.8, 3.3, 3.5, 3.0,
    ];
    const degree = 5;
    const regression = new PolynomialRegression(x, y, degree);
    expect(regression.predict(80)).toBeCloseTo(2.6, 1e-6);
    expect(regression.coefficients).toStrictEqual([
      17.39552328011271, -0.3916378430736305, -0.0019874818431079486,
      0.0001367602062643227, -0.000001302280135149651, 3.837755337564968e-9,
    ]);
    expect(regression.toString(3)).toBe(
      'f(x) = 3.84e-9 * x^5 - 0.00000130 * x^4 + 0.000137 * x^3 - 0.00199 * x^2 - 0.392 * x + 17.4',
    );
  });

  it('toJSON and load', () => {
    const regression = PolynomialRegression.load({
      name: 'polynomialRegression',
      degree: 1,
      powers: [1],
      coefficients: [-1],
    });

    expect(regression.predict(1)).toBe(-1);

    const model = regression.toJSON();
    expect(model).toStrictEqual({
      name: 'polynomialRegression',
      degree: 1,
      powers: [1],
      coefficients: [-1],
    });
  });
  it('Fit a parabola with origin on 0', () => {
    const x = new Float64Array([-4, 4, 2, 3, 1, 8, 5, 7]);
    const y = new Float64Array([16.5, 16.5, 4.5, 9.5, 1.5, 64.5, 25.5, 49.5]);
    const result = new PolynomialRegression(x, y, 2, { interceptAtZero: true });
    const solution = [0.018041553971009705, 1.0095279075485593];
    assertCoefficientsAndPowers(result, solution, [1, 2]);
  });
  it('Fit a parabola with origin on 0, using degree array', () => {
    const x = new Float64Array([-4, 4, 2, 3, 1, 8, 5, 7]);
    const y = new Float64Array([16.5, 16.5, 4.5, 9.5, 1.5, 64.5, 25.5, 49.5]);
    const result = new PolynomialRegression(x, y, [1, 2]);
    const solution = [0.018041553971009705, 1.0095279075485593];
    assertCoefficientsAndPowers(result, solution, [1, 2]);
  });

  it('We should get the same result using numeric degree', () => {
    const x = new Float64Array([-4, 4, 2, 3, 1, 8, 5, 7]);
    // the .5 is to prove that we can force the origin on 0.
    // remove .5 and it tends to y=x^2 as expected.
    const y = new Float64Array([16.5, 16.5, 4.5, 9.5, 1.5, 64.5, 25.5, 49.5]);
    const result = new PolynomialRegression(x, y, 2, {
      interceptAtZero: true,
    });
    const solution = [0.018041553971009705, 1.0095279075485593];
    assertCoefficientsAndPowers(result, solution, [1, 2]);
  });

  it('white noise regression', () => {
    const size = 1000000;
    const x = new Array(size).fill(0).map((_, i) => i);
    const y = Array.from(
      createRandomArray({
        seed: 0,
        mean: 0,
        distribution: 'normal',
        length: size,
      }),
    );
    const regression = new PolynomialRegression(x, y, 1, {
      interceptAtZero: false,
    });
    const newY = [];
    for (let i = 0; i < size; i++) {
      newY.push(y[i] - regression.predict(x[i]));
    }
    const newSumY = xSum(newY);
    expect(newSumY).toBeCloseTo(0, 1e-6);
  });
});
