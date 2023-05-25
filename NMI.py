import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import normalized_mutual_info_score as nmi
import wrangled

def NMI(name, X):
    #binning 1
    X = pd.to_numeric(X)
    X = np.array(X).reshape(-1, 1)
    binner1 = KBinsDiscretizer(n_bins=16, encode='ordinal', strategy='uniform')
    x_bin = binner1.fit_transform(X)
    x_bin = x_bin.flatten()
    #binning 2
    covidRate = np.array(wrangled.covid_rate).reshape(-1, 1)
    binner2 = KBinsDiscretizer(n_bins=16, encode='ordinal', strategy='uniform')
    y_bin = binner2.fit_transform(covidRate)
    y_bin = y_bin.flatten()
    #NMI
    NMI = nmi(x_bin, y_bin, average_method='min')
    print(name, NMI)
    

NMI('Crime rate:', wrangled.crime2021)
NMI('Population Density:', wrangled.populationDensity2020)
NMI('First dose rate:', wrangled.first_dose_rate)
NMI('Second dose rate:', wrangled.second_dose_rate)
NMI('Internal Arrival value:', wrangled.internal_arrival)
