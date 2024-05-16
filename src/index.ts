import { type NumberArray } from 'cheminfo-types';
import { Matrix, MatrixTransposeView, solve } from 'ml-matrix';
import {
  BaseRegression,
  checkArrayLength,
  maybeToPrecision,
} from 'ml-regression-base';

interface PolynomialRegressionOptions {
  interceptAtZero?: boolean;
}
export class PolynomialRegression extends BaseRegression {
  degree: number;
  powers: number[];
  coefficients: number[];
  /**
   * @param x - independent or explanatory variable
   * @param y - dependent or response variable
   * @param degree - degree of the polynomial regression, or array of powers to be used. When degree is an array, intercept at zero is forced to false/ignored.
   * @example `new PolynomialRegression(x, y, 2)`, in this case, you can pass the option `interceptAtZero`, if you need it.
   * @example `new PolynomialRegression(x, y, [1, 3, 5])`
   * Each of the degrees corresponds to a column, so if you have them switched, just do:
   * @example `new PolynomialRegression(x, y, [3, 1, 5])`
   *
   * @param options.interceptAtZero - force the polynomial regression so that f(0) = 0
   */
  constructor(
    x: NumberArray,
    y: NumberArray,
    degree: number | NumberArray,
    options: PolynomialRegressionOptions = {},
  ) {
    super();
    // @ts-expect-error internal use only
    if (x === true) {
      // @ts-expect-error internal use only
      this.degree = y.degree;
      // @ts-expect-error internal use only
      this.powers = y.powers;
      // @ts-expect-error internal use only
      this.coefficients = y.coefficients;
    } else {
      checkArrayLength(x, y);
      const result = regress(x, y, degree, options);
      this.degree = result.degree;
      this.powers = result.powers;
      this.coefficients = result.coefficients;
    }
  }

  _predict(x: number) {
    let y = 0;
    for (let k = 0; k < this.powers.length; k++) {
      y += this.coefficients[k] * x ** this.powers[k];
    }
    return y;
  }

  toJSON() {
    return {
      name: 'polynomialRegression',
      degree: this.degree,
      powers: this.powers,
      coefficients: this.coefficients,
    };
  }

  toString(precision: number) {
    return this._toFormula(precision, false);
  }

  toLaTeX(precision: number) {
    return this._toFormula(precision, true);
  }

  _toFormula(precision: number, isLaTeX: boolean) {
    let sup = '^';
    let closeSup = '';
    let times = ' * ';
    if (isLaTeX) {
      sup = '^{';
      closeSup = '}';
      times = '';
    }

    let fn = '';
    let str = '';
    for (let k = 0; k < this.coefficients.length; k++) {
      str = '';
      if (this.coefficients[k] !== 0) {
        if (this.powers[k] === 0) {
          str = maybeToPrecision(this.coefficients[k], precision);
        } else if (this.powers[k] === 1) {
          str = `${maybeToPrecision(this.coefficients[k], precision) + times}x`;
        } else {
          str = `${
            maybeToPrecision(this.coefficients[k], precision) + times
          }x${sup}${this.powers[k]}${closeSup}`;
        }

        if (this.coefficients[k] > 0 && k !== this.coefficients.length - 1) {
          str = ` + ${str}`;
        } else if (k !== this.coefficients.length - 1) {
          str = ` ${str}`;
        }
      }
      fn = str + fn;
    }
    if (fn.startsWith('+')) {
      fn = fn.slice(1);
    }

    return `f(x) = ${fn}`;
  }

  static load(json: ReturnType<PolynomialRegression['toJSON']>) {
    if (json.name !== 'polynomialRegression') {
      throw new TypeError('not a polynomial regression model');
    }
    // @ts-expect-error internal use only
    return new PolynomialRegression(true, json);
  }
}

/**
 * Perform a polynomial regression on the given data set.
 * This is an internal function.
 * @param x - independent or explanatory variable
 * @param y - dependent or response variable
 * @param degree - degree of the polynomial regression
 * @param options.interceptAtZero - force the polynomial regression so that $f(0) = 0$
 */
function regress(
  x: NumberArray,
  y: NumberArray,
  degree: number | NumberArray,
  options: PolynomialRegressionOptions = {},
) {
  const n = x.length;
  let { interceptAtZero = false } = options;
  let powers: number[] = [];
  if (Array.isArray(degree)) {
    powers = degree;
    interceptAtZero = false; //must be false in this case
  } else if (typeof degree === 'number') {
    if (interceptAtZero) {
      powers = new Array(degree);
      for (let k = 0; k < degree; k++) {
        powers[k] = k + 1;
      }
    } else {
      powers = new Array(degree + 1);
      for (let k = 0; k <= degree; k++) {
        powers[k] = k;
      }
    }
  }
  const nCoefficients = powers.length; //1 per power, in any case.
  const F = new Matrix(n, nCoefficients);
  const Y = new Matrix([y]);
  for (let k = 0; k < nCoefficients; k++) {
    for (let i = 0; i < n; i++) {
      if (powers[k] === 0) {
        F.set(i, k, 1);
      } else {
        F.set(i, k, x[i] ** powers[k]);
      }
    }
  }

  const FT = new MatrixTransposeView(F);
  const A = FT.mmul(F);
  const B = FT.mmul(new MatrixTransposeView(Y));

  return {
    coefficients: solve(A, B).to1DArray(),
    degree: Math.max(...powers),
    powers,
  };
}
