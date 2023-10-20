import { NumberArray } from 'cheminfo-types';
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
});
