import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas.plotting import parallel_coordinates
import wrangled

def drawCovidBarChart():
    # draw bar chart
    plt.bar(np.arange(len(wrangled.old_covid_rate)), wrangled.old_covid_rate)
    plt.xticks(np.arange(len(wrangled.lga)), wrangled.lga, rotation=90)
    plt.tick_params(axis='x', labelsize=3)
    plt.title('Total COVID Cases per 100000 Residents(Until 9/27/2021) for each LGA')
    plt.xlabel('LGA')
    plt.ylabel('Total Covid Cases per 100000 Residents')
    plt.savefig('graph/covidBarChart', dpi=300)
    plt.subplots_adjust(bottom=0.5)
    plt.clf()


def drawCrime_CovidScatter():
    # draw best fit line
    x, y = pd.Series(wrangled.crime2021, name="Crime incidents per 100000 Residents (2021 By March)"), \
           pd.Series(wrangled.covid_rate, name="Total COVID Cases per 100000 Residents")
    sns.regplot(x=x, y=y)
    # draw scatter plot
    plt.scatter(wrangled.crime2021, wrangled.covid_rate)
    plt.title('Compare Crime incidents and Total COVID Cases for each LGA')
    plt.xlabel('Crime incidents per 100000 Residents (2021 By March)', fontsize=10)
    plt.ylabel('Total COVID Cases per 100000 Residents')
    plt.xticks(np.arange(1000, 16000, 1000), rotation=45)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('graph/crime_COVID', dpi=300)
    plt.clf()
            

def drawPopulationDensity_CovidScatter():
    # draw best fit line
    x, y = pd.Series(wrangled.populationDensity2020, name="2020 Population density(person/km2)"), \
           pd.Series(wrangled.covid_rate, name="Total COVID Cases per 100000 Residents")
    sns.regplot(x=x, y=y)
    # draw scatter plot
    plt.scatter(wrangled.populationDensity2020, wrangled.covid_rate)
    plt.title('Compare Population density and Total COVID Cases for each LGA')
    plt.xlabel('2020 Population density(person/km2)', fontsize=10)
    plt.ylabel('Total COVID Cases per 100000 Residents')
    plt.xticks(np.arange(500, 6000, 500), rotation=45)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('graph/populationDensity_COVID', dpi=300)
    plt.clf()


def drawVaccinationRate_CovidScatter():
    # draw first dose best fit line
    x1, y1 = pd.Series(wrangled.first_dose_rate, name="First dose rate(2021 By March)"), \
             pd.Series(wrangled.covid_rate, name="Total COVID Cases per 100000 Residents")
    sns.regplot(x=x1, y=y1)
    # draw first dose scatter plot
    plt.scatter(wrangled.first_dose_rate, wrangled.covid_rate)
    plt.title('Compare first dose rate and Total COVID Cases for each LGA')
    plt.xlabel('First dose rate(2021 By September)', fontsize=10)
    plt.ylabel('Total COVID Cases per 100000 Residents')
    plt.xticks(np.arange(60, 100, 5), rotation=45)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('graph/dose1Rate_covid', dpi=300)
    plt.clf()
    # draw second dose best fit line
    x2, y2 = pd.Series(wrangled.second_dose_rate, name="second dose rate(2021 By March)"), \
             pd.Series(wrangled.covid_rate, name="Total COVID Cases per 100000 Residents")
    sns.regplot(x=x2, y=y2)
    # draw second dose scatter plot
    plt.scatter(wrangled.second_dose_rate, wrangled.covid_rate)
    plt.title('Compare second dose rate and Total COVID Cases for each LGA')
    plt.xlabel('Second dose rate(2021 By September)', fontsize=10)
    plt.ylabel('Total COVID Cases per 100000 Residents')
    plt.ylim(0, 800)
    plt.xticks(np.arange(30, 90, 5), rotation=45)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('graph/dose2Rate_covid', dpi=300)
    plt.clf()


def drawArrivalScatter():
    # draw best fit line
    x, y = pd.Series(wrangled.internal_arrival, name="2019 LGA internal arrival"),\
           pd.Series(wrangled.covid_rate, name="Total COVID Cases per 100000 Residents")
    sns.regplot(x=x, y=y)
    # draw scatter plot
    plt.scatter(wrangled.internal_arrival, wrangled.covid_rate)
    plt.title("Compare Internal arrival and Total COVID Cases for each LGA")
    plt.xlabel("Internal arrival value(2019)", fontsize=10)
    plt.ylabel("Total COVID Cases per 100000 Residents")
    plt.xticks(np.arange(0, 30000, 5000), rotation=45)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('graph/internal_arrival', dpi=300)
    plt.clf()

def drawCorrelation_heatmap():
    # crate dataframe for all varibales
    factors = pd.concat([wrangled.covid_rate, wrangled.crime2021, wrangled.populationDensity2020,
                        wrangled.first_dose_rate, wrangled.second_dose_rate, wrangled.internal_arrival], axis=1)
    factors.columns = ['covid rate', 'crime rate', 'population density', 'first dose rate', 'second dose rate',
                  'internal arrival value']

    # draw correlation Heatmap
    plt.figure(figsize=(15, 9))
    plt.title("Correlations HeatMap", fontsize=30)
    sns.heatmap(abs(factors.corr()), cmap="YlGnBu", annot=True, vmin=0, vmax=1, annot_kws={"fontsize":30})
    plt.savefig('graph/correlation_heatmap', dpi=300)
    plt.clf()

drawCovidBarChart()
drawCrime_CovidScatter()
drawPopulationDensity_CovidScatter()
drawVaccinationRate_CovidScatter()
drawArrivalScatter()
drawCorrelation_heatmap()