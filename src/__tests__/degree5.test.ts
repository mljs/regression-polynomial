import { it, expect } from 'vitest';

import { PolynomialRegression } from '../index';

import { x, y } from './data/degree5.data';

it('degree 5', () => {
  const degree = 5;
  const regression = new PolynomialRegression(x, y, degree);
  expect(regression.predict(80)).toBeCloseTo(2.6, 1e-6);
  expect(regression.coefficients).toStrictEqual([
    20.156354944800594, -0.5790895471877099, 0.003018168825360046,
    0.00007091656373377214, -8.750085815002295e-7, 2.742187260331602e-9,
  ]);
  expect(regression.toLaTeX(5)).toBe(
    'f(x) = 2.7422e-9x^{5} - 8.7501e-7x^{4} + 0.000070917x^{3} + 0.0030182x^{2} - 0.57909x + 20.156',
  );
});
