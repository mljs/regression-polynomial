import { expect, it } from 'vitest';

import { PolynomialRegression } from '..';

import { x, y } from './data/tricky.data';

it('fails', () => {
  const result = new PolynomialRegression(x, y, 2);
  expect(result.degree).toBe(2);
  expect(result.powers).toStrictEqual([0, 1, 2]);
  expect(result.coefficients).toStrictEqual([
    -1315622.3400074367, 6506.470846713673, -8.0445097199281,
  ]);
});
