import numpy as np
import wrangled

global covid_Pearson
covid_Pearson = np.corrcoef(wrangled.covid_rate.to_numpy(), wrangled.covid_rate.to_numpy()).tolist()[0][1]

global crime_Pearson
crime_Pearson =  np.corrcoef(wrangled.crime2021.to_numpy(), wrangled.covid_rate.to_numpy()).tolist()[0][1]
print("The Pearson r between crime rate and covid rate is ", crime_Pearson)

global populationDensity_Pearson
populationDensity_Pearson = np.corrcoef(wrangled.populationDensity2020.to_numpy(), wrangled.covid_rate.to_numpy()).tolist()[0][1]
print("The Pearson r between population density(2020) and covid rate is ", populationDensity_Pearson)

global firstDose_Pearson
firstDose_Pearson = np.corrcoef(wrangled.first_dose_rate, wrangled.covid_rate.to_numpy()).tolist()[0][1]
print("The Pearson r between first dose rate and covid rate is ", firstDose_Pearson)

global secondDose_Pearson
secondDose_Pearson = np.corrcoef(wrangled.second_dose_rate.to_numpy(), wrangled.covid_rate.to_numpy()).tolist()[0][1]
print("The Pearson r between second dose rate and covid rate is ", secondDose_Pearson)

global internalArrival_Pearson
internalArrival_Pearson = np.corrcoef(wrangled.internal_arrival, wrangled.covid_rate.to_numpy()).tolist()[0][1]
print("The Pearson r between internal arrival(2019) and covid rate is ", internalArrival_Pearson)
