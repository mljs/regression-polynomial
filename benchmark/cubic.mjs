/** 
 * Data produced with
  const yVal =
    ((6.02 * (i / 100)) ^ 3) +
    3.01 * (i / 100) ** 2 +
    2.55 * i/100 +
    5 +
    Math.random();
*/

import { readFileSync } from 'fs';
import { join } from 'path';

import { PolynomialRegression } from '../lib/index.js';

const __dirname = new URL('.', import.meta.url).pathname;
const data = JSON.parse(
  readFileSync(
    join(__dirname, '..', 'src', '__tests__', 'data', 'large_cubic.json'),
    'utf8',
  ),
);

/**
 *   const yVal =
    6.3 * (i / 5000) ** 3 +
    3.01 * (i / 1000) ** 2 +
    (2.55 / 100) * i +
    5 +
    Math.random();
 */
const result = new PolynomialRegression(data.x, data.y, 3);

console.log(result.coefficients);
