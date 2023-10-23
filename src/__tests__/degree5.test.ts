import { it, expect } from 'vitest';

import { PolynomialRegression } from '../index';

import { x, y } from './data/degree5.data';
import { assertCoefficientsAndPowers } from './util';

it('degree 5', () => {
  const degree = 5;
  const regression = new PolynomialRegression(x, y, degree);
  expect(regression.predict(80)).toBeCloseTo(2.5625, 5);
  const expectedCs = [
    41451141727940.75, -2814400932400.3633, 75154867680.51097,
    -988576832.4793786, 6415057.648368356, -16448.865765046798,
  ];
  const expectedPowers = [0, 1, 2, 3, 4, 5];
  assertCoefficientsAndPowers(regression, expectedCs, expectedPowers);
  expect(regression.toLaTeX(5)).toBe(
    'f(x) = - 16449x^{5} + 6.4151e+6x^{4} - 9.8858e+8x^{3} + 7.5155e+10x^{2} - 2.8144e+12x + 4.1451e+13',
  );
});
