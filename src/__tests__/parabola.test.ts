import { expect, it } from 'vitest';

import { PolynomialRegression } from '..';

import { x, y } from './data/tricky.data';
import { assertCoefficientsAndPowers } from './util';

it('fit to parabola-like data', () => {
  const result = new PolynomialRegression(x, y, 2);
  const expectedCs = [
    -1126496.3326794421, 5571.151841472432, -6.888109267208182,
  ];
  const expectedPowers = [0, 1, 2];
  assertCoefficientsAndPowers(result, expectedCs, expectedPowers);
  expect(result.degree).toBe(2);
});
