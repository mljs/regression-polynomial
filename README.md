# regression-polynomial

[![NPM version][npm-image]][npm-url]
[![npm download][download-image]][download-url]
[![build status][ci-image]][ci-url]
[![Test coverage][codecov-image]][codecov-url]

Polynomial Regression.

## Installation

`$ npm i ml-regression-polynomial`

## Usage

```js
import { PolynomialRegression } from 'ml-regression-polynomial';

const x = [50, 50, 50, 70, 70, 70, 80, 80, 80, 90, 90, 90, 100, 100, 100];
const y = [
  3.3, 2.8, 2.9, 2.3, 2.6, 2.1, 2.5, 2.9, 2.4, 3.0, 3.1, 2.8, 3.3, 3.5, 3.0,
];
const degree = 5; // setup the maximum degree of the polynomial

const regression = new PolynomialRegression(x, y, degree);

console.log(regression.predict(80)); // Apply the model to some x value. Prints 2.6.
console.log(regression.coefficients); // Prints the coefficients in increasing order of power (from 0 to degree).
console.log(regression.toString(3)); // Prints a human-readable version of the function.
console.log(regression.toLaTeX());
console.log(regression.score(x, y));
```

## Options

An `interceptAtZero` option is available, to force $f(0) = 0$. Also, a "powers array" can be specified.

- Using `interceptAtZero`

```js
const regression = new PolynomialRegression(x, y, degree, {
  interceptAtZero: true,
});
```

- Using the powers array

```js
const powers = [0, 1, 2, 3, 4, 5];
const regression = new PolynomialRegression(x, y, powers);
```

`powers` could also be `[1,2,3,4,5]`or`[1,3,5]` and so on.

For intercepting at zero using an array, skip the zero in the array (the option `interceptAtZero` is ignored in this case.)

## License

[MIT](./LICENSE)

[npm-image]: https://img.shields.io/npm/v/ml-regression-polynomial.svg?style=flat-square
[npm-url]: https://npmjs.org/package/ml-regression-polynomial
[download-image]: https://img.shields.io/npm/dm/ml-regression-polynomial.svg?style=flat-square
[download-url]: https://npmjs.org/package/ml-regression-polynomial
[codecov-image]: https://img.shields.io/codecov/c/github/mljs/regression-polynomial.svg
[codecov-url]: https://codecov.io/gh/mljs/regression-polynomial
[ci-image]: https://github.com/mljs/regression-polynomial/workflows/Node.js%20CI/badge.svg?branch=main
[ci-url]: https://github.com/mljs/regression-polynomial/actions?query=workflow%3A%22Node.js+CI%22
