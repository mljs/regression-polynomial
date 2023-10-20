import { readFileSync } from 'fs';
import { join } from 'path';

import { PolynomialRegression } from '../lib/index.js';

const __dirname = new URL('.', import.meta.url).pathname;
const data = JSON.parse(
  readFileSync(
    join(__dirname, '..', 'src', '__tests__', 'data', 'large_quadratic.json'),
    'utf8',
  ),
);

/**
 *
 *   const yVal = 3.01 * (i / 100) ** 2 + (2.55/100) * i + 5 + Math.random();
 */
const result = new PolynomialRegression(data.x, data.y, 2);

// console.log(result);
