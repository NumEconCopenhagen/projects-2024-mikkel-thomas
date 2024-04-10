import pandas as pd

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

    def __init__(self, df):
        self.data = df
        self.gpr = self.data.copy()
        self.gpr_long = self.drop_cols()
        self.gpr_long = self.country_lookup()
        self.gpr_long = self.stadardize_gpr_apply()

    def drop_cols(self):
        # Selecting the columns with the country specific historical GPR index
        gpr = self.gpr
        gpr = gpr[[col for col in gpr.columns if col.startswith('GPRHC_')]]

        # Melt the DataFrame to long format
        gpr_long = gpr.reset_index().melt(id_vars='Date', var_name='Country', value_name='GPRH')
        gpr_long.set_index('Date', inplace=True)
        
        return gpr_long

    def country_lookup(self):
        filename = 'country_lookup.xlsx'
        country_lookup = pd.read_excel(filename)

        gpr_long = self.gpr_long.reset_index().merge(country_lookup, on='Country', how='left').set_index(['Date', 'Country_name'])
        gpr_long.reset_index(inplace=True)
        gpr_long.drop(columns='Country', inplace=True)

        return gpr_long
    
    def standardize_gpr(self,df, country_col, gpr_col):
        df[gpr_col] = df.groupby(country_col)[gpr_col].transform(lambda x: (x - x.mean()) / x.std())
        return df

    def stadardize_gpr_apply(self):
        gpr_long = self.standardize_gpr(self.gpr_long, 'Country_name', 'GPRH')
        
        return gpr_long

    def __call__(self):
        # Return the cleaned DataFrame when the instance is called like a function
        return self.gpr_long
    
class idealpoint_clean():

    def __init__(self, df):
        self.df = df
        self.long_df_q = self.clean_idealpoint()

    def clean_idealpoint(self):
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