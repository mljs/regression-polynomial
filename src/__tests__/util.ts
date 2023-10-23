import { NumberArray } from 'cheminfo-types';
import { expect } from 'vitest';

import { PolynomialRegression } from '..';

/**
 * Compare the coefficients and powers of a PolynomialRegression instance with the expected values, to the sixth decimal place.
 * @param result - The PolynomialRegression result.
 * @param expectedCs - Coefficients in the expected order.
 * @param expectedPowers - Powers in the expected order.
 * @param numDigits - Number of digits to compare to, defaults to 6.
 */
export function assertCoefficientsAndPowers(
  result: PolynomialRegression,
  expectedCs: NumberArray,
  expectedPowers: NumberArray,
  numDigits = 6,
) {
  for (let i = 0; i < expectedCs.length; ++i) {
    expect(result.coefficients[i]).toBeCloseTo(expectedCs[i], numDigits || 6);
  }
  expect(result.powers).toStrictEqual(expectedPowers);
  expect(result.degree).toBe(Math.max(...expectedPowers));
}
