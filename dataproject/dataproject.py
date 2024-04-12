import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pycountry_convert as pc
from pandas.api.types import CategoricalDtype
from joypy import joyplot
from matplotlib import cm
import altair as alt

def keep_regs(df, regs):
    """ Example function. Keep only the subset regs of regions in data.

    Args:
        df (pd.DataFrame): pandas dataframe 

    Returns:
        df (pd.DataFrame): pandas dataframe

    """ 
    
    for r in regs:
        I = df.reg.str.contains(r)
        df = df.loc[I == False] # keep everything else
    
    return df

class gpr_clean():
    """
    Class to clean the Geopolitical Risk Index (GPR) data
    """

    def __init__(self, df):
        """
        Initialize the class with the DataFrame to be cleaned

        Args:
            df (pd.DataFrame): DataFrame to be cleaned
        Returns:
            None
        """
        self.data = df
        self.gpr = self.data.copy()
        self.gpr_long = self.drop_cols()
        self.gpr_long = self.country_lookup()
        self.gpr_long = self.stadardize_gpr_apply()

    def drop_cols(self):
        """
        Drop columns that are not needed for the analysis

        Returns:
            gpr_long (pd.DataFrame): DataFrame with the relevant columns
        """

        # Selecting the columns with the country specific historical GPR index
        gpr = self.gpr
        gpr = gpr[[col for col in gpr.columns if col.startswith('GPRHC_')]]

        # Melt the DataFrame to long format
        gpr_long = gpr.reset_index().melt(id_vars='Date', var_name='Country', value_name='GPRH')
        gpr_long.set_index('Date', inplace=True)
        
        return gpr_long

    def country_lookup(self):
        """ 
        Merge the DataFrame with the country lookup table that maps the original variable names to the full country names

        Returns:
            gpr_long (pd.DataFrame): DataFrame with the full country names and the GPR index
        """
        filename = 'country_lookup.xlsx'
        country_lookup = pd.read_excel(filename)

        gpr_long = self.gpr_long.reset_index().merge(country_lookup, on='Country', how='left').set_index(['Date', 'Country_name'])
        gpr_long.reset_index(inplace=True)
        gpr_long.drop(columns='Country', inplace=True)

        return gpr_long
    
    def standardize_gpr(self,df, country_col, gpr_col):
        """
        Standardize the GPR index by country. This function is used in the 'stadardize_gpr_apply' method
        Args:
            df (pd.DataFrame): DataFrame to be cleaned
            country_col (str): Name of the column with the country names
            gpr_col (str): Name of the column with the GPR index
        Returns:
            df (pd.DataFrame): DataFrame with the standardized GPR index as a column
        """

        df[gpr_col] = df.groupby(country_col)[gpr_col].transform(lambda x: (x - x.mean()) / x.std())
        return df

    def stadardize_gpr_apply(self):
        """
        Standardize the GPR index by country
        Args:
            None
        Returns:
            gpr_long (pd.DataFrame): DataFrame with the standardized GPR index
        """
        gpr_long = self.standardize_gpr(self.gpr_long, 'Country_name', 'GPRH')
        
        gpr_long['Date'] = gpr_long['Date'].dt.date
        return gpr_long

    def __call__(self):
        """"
        Return the cleaned DataFrame when the instance is called like a function
        """
        # Return the cleaned DataFrame when the instance is called like a function
        return self.gpr_long
    
class idealpoint_clean():

    def __init__(self, df):
        """
        Initialize the class with the DataFrame to be cleaned
        Args:
            df (pd.DataFrame): DataFrame to be cleaned
        Returns:
            None
        """
        self.df = df
        self.long_df_q = self.clean_idealpoint()

    def clean_idealpoint(self):
        """
        Function to clean the ideal point data
        Args:
            None
        Returns:
            long_df_q (pd.DataFrame): DataFrame with the ideal point data in long format and resampled to quarterly frequency
        """
        pivot_df = self.df.pivot_table(index='session', columns='Countryname', values='IdealPointAll')
        pivot_df.columns.name = None
    
        pivot_df['ideal_left'] = pivot_df[pivot_df < 0].mean(axis=1)
        pivot_df['ideal_right'] = pivot_df[pivot_df > 0].mean(axis=1)
        pivot_df.index = pd.date_range(start='01-01-1946', periods=len(pivot_df), freq='AS')
        long_df = pivot_df.reset_index()
        long_df = pivot_df.reset_index().melt(id_vars='index', var_name='Countryname', value_name='IdealPointAll')

        long_df = long_df.rename(columns={'Countryname': 'Country_name', 'index': 'Date'})
        long_df.set_index('Date', inplace=True)
        
        long_df_q = long_df.groupby('Country_name').resample('Q').ffill()
        long_df_q.drop(columns='Country_name', inplace=True)

        # Convert 'Date' to Period with quarterly frequency, then convert back to timestamps at the start of the quarter
        long_df_q.reset_index(inplace=True)
        long_df_q['Date'] = long_df_q['Date'].dt.to_period('Q').dt.to_timestamp()

        # Set 'Date' and 'Country_name' back as the index
        long_df_q.set_index(['Date', 'Country_name'], inplace=True)

        return long_df_q
    
    def __call__(self):
        # Return the cleaned DataFrame when the instance is called like a function
        return self.long_df_q
    
def heatmap_gpr(df_gpr):
    df_heatmap = df_gpr.pivot_table(index='Country_name', columns='Date', values='GPRH')

    # Selecting data from 1990 to the present
    df_heatmap = df_heatmap.loc[:, datetime.date(1990, 1, 1):]

    # Flatten the data
    data_flat = df_heatmap.values.flatten()

    # Calculate the 2st and 98th percentiles
    vmin = np.percentile(data_flat, 2)
    vmax = np.percentile(data_flat, 98)

    plt.figure(figsize=(14, 10))
    heatmap = sns.heatmap(df_heatmap, cmap='coolwarm', vmin=vmin, vmax=vmax)

    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Geopolitical Risk Index (GPRH)', fontsize=13)

    # New code to modify x-tick labels
    ax = plt.gca()
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=45, horizontalalignment='right')

    # Get current x-tick locations and labels
    locs, labels = plt.xticks()

    # Set x-tick locations and labels
    new_locs = locs[::4]  # Select every 5th location
    new_labels = [labels[i].get_text()[:4] for i in range(len(labels)) if i % 4 == 0]  # Select every 5th label

    plt.xticks(new_locs, new_labels)

    ax.set_ylabel('Country', fontsize=13)
    ax.set_xlabel('Date', fontsize=13)
    title = ax.set_title('Country Level Geopolitical Risk Index (1990-2023)', fontsize=13)
    # title.set_position([-.18, 1])
    plt.show()

def lineplot_idealpoints(df_idealpoint):

    pivot_df = df_idealpoint.pivot_table(index='Date', columns='Country_name', values='IdealPointAll')
    pivot_df.columns.name = None
    pivot_df = pivot_df.resample('A').mean()
    # Plot the underlying 'IdealPointAll' lines in grey
    plt.figure(figsize=(12,8))
    for column in pivot_df.columns:
        if column not in ['ideal_left', 'ideal_right']:
            plt.plot(pivot_df.index, pivot_df[column], color='grey', alpha=0.2)

    # Plot the 'ideal_left' and 'ideal_right' lines in different colors
    # plt.plot(pivot_df.index, pivot_df['United States'], label='United States')
    # plt.plot(pivot_df.index, pivot_df['China'], label='China')
    plt.plot(pivot_df.index, pivot_df['ideal_left'], color='blue', label='Average Ideal Point (left-leaning)')
    plt.plot(pivot_df.index, pivot_df['ideal_right'], color='red', label='Average Ideal Point (right-leaning)')

    # Setting range of x-axis
    plt.xlim([datetime.date(1946, 12, 31), datetime.date(2021, 12, 31)])

    plt.grid(True, linestyle = '--', alpha = 0.5)
    plt.xlabel('Year', fontsize = 13)
    plt.ylabel('Ideal Point', fontsize = 13)
    plt.title('Ideal Point Estimates Based on UN Voting Behaviour', fontsize = 13)
    plt.legend()
    plt.show()

def scatter_mil_ideal(df_military_gdp_spend, df_population, df_idealpoint):
    df_military_gdp_spend = df_military_gdp_spend[['2022']]
    df_military_gdp_spend.reset_index(inplace=True)
    df_military_gdp_spend = df_military_gdp_spend.rename(columns={'Country Code': 'iso3c', '2022': 'Military_GDP_Spend'})

    df_population = df_population[['2022']]
    df_population.reset_index(inplace=True)
    df_population = df_population.rename(columns={'Country Code': 'iso3c', '2022': 'population_level'})

    df_idealpoint = df_idealpoint[['iso3c', 'IdealPointAll', 'session']]

    # Keep only idealpoint from session 76 (i.e. 2022)
    df_idealpoint = df_idealpoint[df_idealpoint['session'] == 76]

    merged_df = df_idealpoint.reset_index().merge(df_military_gdp_spend, left_on = 'iso3c', right_on = 'iso3c', how='inner')
    merged_df = merged_df.reset_index().merge(df_population, left_on = 'iso3c', right_on = 'iso3c', how='inner')
    merged_df.drop(columns=['index', 'session', 'level_0'], inplace=True)
    merged_df.set_index('iso3c', inplace=True)
    merged_df.dropna(inplace=True)

    def iso3c_to_continent(iso3c):
        try:
            country_alpha2 = pc.country_alpha3_to_country_alpha2(iso3c)
            country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
            country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
            return country_continent_name
        except KeyError:
            return np.nan

    def add_continent_to_df(df):
        df['Continent'] = df.index.map(iso3c_to_continent)
        return df

    merged_df = add_continent_to_df(merged_df)

    # Scatter plot of military spend vs idealpoint
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=merged_df, x='IdealPointAll', y='Military_GDP_Spend', size = 'population_level', sizes=(20, 2000), hue='Continent', palette='tab10', alpha=0.7)

    # Add grid lines
    plt.grid(True, linestyle = '--', alpha = 0.5)

    # Add labels for specific points
    high_pop_countries = merged_df[merged_df['population_level'] > 100*10**6]
    high_pop_countries_index = high_pop_countries.index
    for country in high_pop_countries_index:
        country_data = merged_df.loc[country]
        plt.text(x=country_data['IdealPointAll'], 
                y=country_data['Military_GDP_Spend'], 
                s=country, 
                fontdict=dict(color='black',size=12),
                bbox=dict(facecolor='yellow',alpha=0.))

    plt.xlabel('Ideal point', fontsize=16)
    plt.ylabel('Military spending, percent of GDP', fontsize=16)
    plt.title('The left-leaning block vastly outweigh the right-leaning in terms of population...', fontsize=16)
    plt.ylim(0, 9)

    plt.show()

def ridge_gpr(df_gpr):

    def windsorize_series(s, lower_quantile=0.01, upper_quantile=0.99):
        lower = s.quantile(lower_quantile)
        upper = s.quantile(upper_quantile)
        return s.clip(lower, upper)

    # Create a list of decades from 1900 to 2090
    decades = [str(i) for i in range(1900, 2025, 5)]

    # Define the categorical data type
    cat_decade = CategoricalDtype(decades)

    df_gpr_ridge = df_gpr.copy()

    # Calculate the mean of the 'GPRH' by date
    df_gpr_ridge['GPRH'] = df_gpr_ridge.groupby('Date')['GPRH'].transform('mean')

    # Drop duplicates by date
    df_gpr_ridge = df_gpr_ridge.drop_duplicates(subset='Date')

    df_gpr_ridge['GPRH_Windsorized'] = df_gpr_ridge.groupby('Country_name')['GPRH'].transform(windsorize_series)

    # Convert the 'Date' column to datetime if it's not already
    df_gpr_ridge['Date'] = pd.to_datetime(df_gpr_ridge['Date'])

    # Extract the year from the 'Date' column
    df_gpr_ridge['Year'] = df_gpr_ridge['Date'].dt.year

    # Create a new column for the decade
    df_gpr_ridge['Decade'] = (df_gpr_ridge['Year'] // 5) * 5
    df_gpr_ridge['Decade'] = df_gpr_ridge['Decade'].astype(str).astype(cat_decade)

    # Dropping all rows not equal to argentina or united states
    df_gpr_ridge = df_gpr_ridge[df_gpr_ridge['Country_name'].isin(['Argentina', 'United States'])]
    # df_gpr_ridge = df_gpr_ridge[df_gpr_ridge['Country_name'] == 'Argentina']
    df_gpr_ridge = df_gpr_ridge.drop(columns=['Country_name', 'Date', 'Year', 'GPRH'])
    df_gpr_ridge.head()

    plt.figure()

    joyplot(
        data=df_gpr_ridge[['GPRH_Windsorized', 'Decade']], 
        by='Decade',
        alpha=0.85,
        colormap=cm.coolwarm,
        figsize=(12, 8)
    )
    plt.title('Distribution of Global GPR index for Five-Year Periods (1900-2023)', fontsize=13)
    plt.xlabel('Average Geopolitical Risk Index, all countries', fontsize=13)
    plt.show()

def lineplot_war_deaths(df_war_deaths):
    df_war_deaths = df_war_deaths[df_war_deaths['Entity'] == 'World']

    df_war_deaths = df_war_deaths.drop(columns=['Code', 'Entity'])
    df_war_deaths = df_war_deaths.melt(id_vars='Year', var_name='Type', value_name='Count')
    # Converting the year column to datetime
    df_war_deaths['Year'] = pd.to_datetime(df_war_deaths['Year'], format='%Y')

    plot = alt.Chart(df_war_deaths).mark_circle(
    opacity=0.8,
    stroke='black',
    strokeWidth=1,
    strokeOpacity=0.4
    ).encode(
    alt.X('Year:T')
        .title(None)
        .scale(domain=['1989','2022']),
    alt.Y('Type:N')
        .title(None)
        .sort(field="Count", op="sum", order='descending'),
    alt.Size('Count:Q')
        .scale(range=[0, 2500])
        .title('Deaths')
        .legend(clipHeight=30, format='s'),
    alt.Color('Type:N').legend(None),
    tooltip=[
        "Type:N",
        alt.Tooltip("Year:T", format='%Y'),
        alt.Tooltip("Count:Q", format='~s')
    ],
    ).properties(
    width=1.3*450,
    height=320,
    title=alt.Title(
        text="Global Deaths from State-Based Wars (1989-2022)",
        subtitle="The size of the bubble represents the total death count per year, by type of war",
        anchor='start'
    )
    ).configure_axisY(
    domain=False,
    ticks=False,
    offset=10
    ).configure_axisX(
    grid=False,
    ).configure_view(
    stroke=None
    )

    return plot

# OLD CODE BELOW - NOT USED IN THE FINAL PROJECT

class dots_clean():

    def __init__(self, df):
        self.df = df
        self.dots_imf = self.df.copy()
        self.dots_imf = self.fill_nan()
        self.dots_imf = self.new_vars()
        self.dots_imf = self.new_names()
        self.dots_imf = self.slim_data()

    def fill_nan(self): 
        dots_imf = self.dots_imf.fillna(0)

        return dots_imf
    
    def new_vars(self):
        dots_imf = self.dots_imf
        dots_imf['Trade'] = self.dots_imf['Export'] + self.dots_imf['Import']

        dots_imf['tot_trade'] = dots_imf.groupby(['Country_Name', 'Date'])['Trade'].transform('sum')
        dots_imf['trade_share'] = dots_imf['Trade'] / dots_imf['tot_trade']
        dots_imf['check'] = dots_imf.groupby(['Country_Name', 'Date'])['trade_share'].transform('sum')
        dots_imf['Date'] = pd.to_datetime(dots_imf['Date']).dt.to_period('Q').dt.start_time
        dots_imf = dots_imf.rename(columns={'Country_Name': 'imf_name'})
        dots_imf = dots_imf.rename(columns={'Counterpart_Country_Name': 'imf_name_counterpart'})

        return dots_imf
    
    def new_names(self):

        dots_imf = self.dots_imf
        # Merging the DataFrame with the lookup tables
        filename = 'imf_name_lookup.xlsx'
        imf_lookup = pd.read_excel(filename)
        filename = 'imf_name_cp_lookup.xlsx'
        imf_lookup_cp = pd.read_excel(filename)
        dots_imf = dots_imf.reset_index().merge(imf_lookup, on='imf_name', how='left')
        dots_imf = dots_imf.reset_index().merge(imf_lookup_cp, on='imf_name_counterpart', how='left')

        # Rename the columns
        dots_imf = dots_imf.rename(columns={'Country_name_x': 'country'})
        dots_imf = dots_imf.rename(columns={'Country_name_y': 'counterpart'})
        # dots_imf.head(5)
        return dots_imf
    
    def slim_data(self):
        # Select the relevant columns
        dots_imf = self.dots_imf[['Date','country', 'counterpart', 'trade_share']]
        dots_imf = dots_imf.rename(columns={'country': 'Country_name', 'counterpart': 'Counterpart_name'})

        dots_imf.set_index(['Date', 'Country_name'], inplace=True)
        dots_imf = dots_imf.sort_values(['Country_name', 'Date'])

        return dots_imf
    
    def __call__(self):
        # Return the cleaned DataFrame when the instance is called like a function
        return self.dots_imf
    
class merge_dots_gpr():

    def __init__(self, df_dots, df_gpr):
        self.df_dots = df_dots
        self.df_gpr = df_gpr

        self.gpr_trade = self.run()

    def merge_data(self):
        df_dots = self.df_dots
        df_gpr = self.df_gpr
        merged_df = df_dots.reset_index().merge(df_gpr, on=['Country_name', 'Date'], how='left')
        merged_df = merged_df.rename(columns={'Country_name': 'name', 'Counterpart_name': 'Country_name'})

        merged_df_2 = merged_df.reset_index().merge(df_gpr, on=['Country_name', 'Date'], how='left')

        return merged_df_2
    
    def trade_weighted_gpr(self):

        merged_df_2 = self.merged_df_2
        merged_df_2['product'] = merged_df_2['trade_share'] * merged_df_2['GPRH_y']
        merged_df_2['sumproduct'] = merged_df_2.groupby(['Date', 'name'])['product'].transform('sum')
        merged_df_2['GPRH_trade_weight'] = merged_df_2['sumproduct'] + merged_df_2['GPRH_x']

        merged_df_2_no_dub = merged_df_2.drop_duplicates(subset=['Date', 'name', 'GPRH_trade_weight'])

        return merged_df_2_no_dub

    def slim_data(self):
        merged_df_2_no_dub = self.merged_df_2_no_dub
        gpr_trade = merged_df_2_no_dub[['Date','name', 'GPRH_trade_weight']]
        gpr_trade.set_index(['Date', 'name'], inplace=True)
        gpr_trade.reset_index(inplace=True)

        gpr_trade = gpr_trade.rename(columns={'name': 'Country_name'})

        return gpr_trade
    
    def run(self):
        self.merged_df_2 = self.merge_data()
        self.merged_df_2_no_dub = self.trade_weighted_gpr()
        self.gpr_trade = self.slim_data()

        return self.gpr_trade
    
    def __call__(self):
        # Return the cleaned DataFrame when the instance is called like a function
        return self.gpr_trade
    
class merge_data():

    def __init__(self, df_gpr, df_idealpoint):
        self.df_gpr = df_gpr
        self.df_idealpoint = df_idealpoint

        self.merged_df = self.merge_data()

    def merge_data(self):
        merged_df = self.df_gpr.reset_index().merge(self.df_idealpoint, on=['Country_name', 'Date'], how='left')
        merged_df = merged_df[merged_df['Date'].dt.year >= 1946]
        merged_df.drop(columns='index', inplace=True)
        
        merged_df.dropna(inplace=True)
        merged_df.set_index(['Date', 'Country_name'], inplace=True)

        # Calculate the average of 'IdealPointAll' by 'Country_name'
        average_ideal_point = merged_df.groupby(level='Country_name')['IdealPointAll'].mean()

        # Create a new column that assigns 1 if the average of 'IdealPointAll' is positive and 0 otherwise
        merged_df['NewColumn'] = merged_df.index.get_level_values('Country_name').map(lambda x: 1 if average_ideal_point[x] > 0 else 0)
        
        return merged_df
    
    def __call__(self):
        # Return the cleaned DataFrame when the instance is called like a function
        return self.merged_df