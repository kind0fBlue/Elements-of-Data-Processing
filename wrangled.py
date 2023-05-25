import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

covidPath = "raw_data/COVID_data.csv"
crimePath = "raw_data/CrimeIncident.csv"
populationDensityPath = "raw_data/PopulationDensity.csv"
vaccinationRatePath = "raw_data/Vaccinations_LGA.csv"
absPath = "raw_data/ABS4.0.csv"


global covid_rate
global lga
global old_covid_rate #variable added by ziming, for visualization only not for calculation
covid = pd.read_csv(covidPath, encoding='ISO-8859-1')
lga = covid['LGADisplay']
covid_rate = 100000 / covid['population'] * covid['cases']
old_covid_rate = covid_rate.copy()

global crime2021
# read and pre processing data
crime = pd.read_csv(crimePath, encoding='ISO-8859-1')
crime = crime.sort_values(by='Local Government Area')
crime2021 = crime['2021']
crime2021 = pd.to_numeric(crime2021)

global populationDensity2020
popualtionDensity = pd.read_csv(populationDensityPath, encoding='ISO-8859-1')
popualtionDensity['Local Government Area'] = popualtionDensity['Local Government Area'].replace('\([a-zA-Z]*\)', '', regex=True)
populationDensity = popualtionDensity.sort_values(by='Local Government Area')
populationDensity2020 = populationDensity['Population density 2020(persons/km2)']
populationDensity2020 = pd.to_numeric(populationDensity2020)

global first_dose_rate
global second_dose_rate
vaccinationRate = pd.read_csv(vaccinationRatePath)
vaccinationRate = vaccinationRate.sort_values(by='LGA')
first_dose_rate = vaccinationRate["FIRST"]
second_dose_rate = vaccinationRate["SECOND"]

global internal_arrival
abs = pd.read_csv(absPath, encoding='ISO-8859-1')
df = pd.DataFrame({"LGA": abs["Region"], "Data type": abs["Data item"], "Time": abs["TIME"], "Internal_Arrival_Value": abs["Value"]})
df2 = df.loc[df['Time'] == 2019].loc[df['Data type'] == "Internal Arrivals (no.)"]
covid = pd.read_csv(covidPath, encoding='ISO-8859-1')
lga = covid["LGA"]
covid_rate = 100000 / covid['population'] * covid['cases']
df3 = pd.DataFrame({"LGA": lga, "Cases/100000": covid_rate})
df4 = pd.merge(df2, df3, on='LGA')
internal_arrival = df4["Internal_Arrival_Value"]


####################### Ziming || remove outlier##############################
#visualize outliers
plt.boxplot(covid_rate)
plt.title('Visualising covid')
plt.ylabel('covid cases per 100000 residents')
plt.tight_layout()
plt.savefig('graph/boxplot of covid', dpi=300)
plt.clf()
#remove all value bigger than 700
outlierIndex = []
for i in range(0, 79):
    if covid_rate[i] > 800:
        outlierIndex.append(i)
#change series to list to eliminate index disorder we didn't follow the same convention
covid_rate = list(covid_rate)
crime2021 = list(crime2021)
populationDensity2020 = list(populationDensity2020)
first_dose_rate = list(first_dose_rate)
second_dose_rate = list(second_dose_rate)
internal_arrival = list(internal_arrival)

cpnewCovid= covid_rate.copy()
cpnewCrime= crime2021.copy()
cpPopulation = populationDensity2020.copy()
cpnewfirstDose= first_dose_rate.copy()
cpnewSecondDose= second_dose_rate.copy()
cpnewTravel= internal_arrival.copy()

for index in outlierIndex:
    covid_rate.remove(cpnewCovid[index])
    crime2021.remove(cpnewCrime[index])
    populationDensity2020.remove(cpPopulation[index])
    first_dose_rate.remove(cpnewfirstDose[index])
    second_dose_rate.remove(cpnewSecondDose[index])
    internal_arrival.remove(cpnewTravel[index])
#change the list back to series to rebuild the index
covid_rate = pd.Series(covid_rate)
crime2021 = pd.Series(crime2021)
populationDensity2020 = pd.Series(populationDensity2020)
first_dose_rate = pd.Series(first_dose_rate)
second_dose_rate = pd.Series(second_dose_rate)
internal_arrival = pd.Series(internal_arrival)


