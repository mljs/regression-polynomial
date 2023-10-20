import React from 'react';
import { PolynomialRegression } from '../src/index';
import {
  Plot,
  LineSeries,
  Axis,
  Legend,
  Heading,
  SeriesPoint,
} from 'react-plot';

import { x, y } from '../src/__tests__/data/degree5.data';

const data = x.map((x, i) => ({ x, y: y[i] }));

const calculations: SeriesPoint[][] = [];
const degree: number[] = [];
for (let i = 2; i <= 7; i++) {
  const r = new PolynomialRegression(x, y, i)
    if(i===5) console.log(r.coefficients, r.toLaTeX(5))
    
  const plot = r.predict(x).map((y, i) => ({ x: x[i], y }));
  degree.push(r.degree);
  calculations.push(plot);
}

export const Example = () => (
  <Plot
    width={1000}
    height={1000}
    margin={{ bottom: 50, left: 90, top: 50, right: 100 }}
  >
    <Heading
      title="Electrical characterization"
      subtitle="Current vs Voltage"
    />
    <LineSeries
      data={data}
      xAxis="x"
      yAxis="y"
      lineStyle={{ strokeWidth: 3 }}
      label="Raw Data"
      displayMarkers={false}
    />
    {calculations.map((plot, i) => (
      <LineSeries
        key={i}
        data={plot}
        xAxis="x"
        yAxis="y"
        label={`${degree[i]} degree polynomial`}
      />
    ))}
    <Axis
      id="x"
      position="bottom"
      label="Drain voltage [V]"
      displayPrimaryGridLines
      // max={Math.max(...x) * 1.1}
    />
    <Axis
      id="y"
      position="left"
      label="Drain current [mA]"
      displayPrimaryGridLines
      // max={Math.max(...y) * 1.1}
    />
    <Legend position="right" />
  </Plot>
);
