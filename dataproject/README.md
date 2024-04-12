# Data analysis project

Our project is titled **Have the World Become Less Peaceful? An Emirical Analysis of Geopolitical Trends** and is about the development of war and peach over time in the world.

The results of the project can be seen from running [dataproject.ipynb](dataproject.ipynb). This file will generate five different charts each telling a piece of the story about war and peace in the world.

The respositiory also contains the dataproject.py file, which contains classes and functions to clean and merge data. The folder `misc` contain old stuff and should not be used.

We apply the **following datasets**:

1. GPR_quarter.xlsx ([Caldara and Iacoviello (2022)](https://www.matteoiacoviello.com/gpr.htm))
1. IdealpointestimatesAll_Sep2023.csv ([Bailey, Strezhnev and Voeten (2017)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LEJUQZ))
1. military_gdp_spend.csv ([World Bank WDI](https://data.worldbank.org/indicator/MS.MIL.XPND.GD.ZS))
1. population.csv ([World Bank WDI](https://data.worldbank.org/indicator/SP.POP.TOTL))
1. war_deaths.csv ([Our World in Data](https://ourworldindata.org/explorers/conflict-data?facet=none&Conflict+type=All+armed+conflicts&Measure=Conflict+deaths&Conflict+sub-type=By+sub-type&Data+source=Uppsala+Conflict+Data+Program&Sub-measure=Regional+data&country=~OWID_WRL))

In addition to the above, we also use two lookup tables to match varible names to countries names (country_lookup.xlsx).

The project is structured as follows:
1. `dataproject.ipynb` contains all our charts
1. `dataproject.py` does the heavy lifting in the form of data cleaning and plotting 


**Dependencies:** Apart from a standard Anaconda Python 3 installation, the project requires the following installations:

``pip install matplotlib-venn``

``pip install joypy``

``pip install pycountry_convert``
