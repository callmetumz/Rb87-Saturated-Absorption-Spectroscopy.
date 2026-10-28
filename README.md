# Saturated absorption spectroscopy of Rubidium-87

Data analysis for PHY324 Project 1 (University of the Western Cape), by Tumelo Aphane.

The measurements were taken in the laser laboratory at the Cape Peninsula University of Technology (CPUT) on 3 August 2026, using a pump-probe saturated absorption setup on a natural rubidium vapour cell.

## What the analysis does

1. Plots the full sweep of the D2 line with the pump on and blocked, and estimates the laser drift between the two captures.
2. Fits each 87Rb hyperfine spectrum (F = 2 and F = 1, three runs each) with a Gaussian Doppler background plus six Lorentzian peaks.
3. Calibrates the oscilloscope time axis to frequency with a linear fit against the literature peak frequencies (Steck, Rubidium 87 D Line Data).
4. Reports the peak frequencies, the excited-state hyperfine spacings and the linewidths, including the cooling (F = 2 to F' = 3) and repump (F = 1 to F' = 2) transitions.

## Files

| File | Content |
|---|---|
| `PHY324_Project1_Rb87_Data_Analysis.ipynb` | The analysis notebook |
| `PHY324_Project1_Rb87_Data_Analysis.py` | The same code as a plain Python script |
| `F0000CH1.CSV`, `F0000CH2.CSV` | Full scan, pump on (ramp and photodetector signal) |
| `F0001CH1.CSV`, `F0001CH2.CSV` | Full scan, pump blocked |
| `F0002CH2.CSV` to `F0004CH2.CSV` | 87Rb F = 2 spectrum, three runs |
| `F0005CH2.CSV` to `F0007CH2.CSV` | 87Rb F = 1 spectrum, three runs |

The CSV files are Tektronix TDS2024B oscilloscope captures (2500 points, 10 µs per point).

## Running it

Requires Python 3 with numpy, scipy, pandas and matplotlib. Open the notebook in Jupyter and run all cells. The data files must be in the same folder as the notebook. Figures are saved to `figures/` and the result tables to CSV files in the same folder.
