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

    def drop_cols(self):
        # Selecting the columns with the country specific historical GPR index
        gpr = self.gpr
        gpr = gpr[[col for col in gpr.columns if col.startswith('GPRHC_')]]

        # Melt the DataFrame to long format
        gpr_long = gpr.reset_index().melt(id_vars='Date', var_name='Country', value_name='GPRH')
        gpr_long.set_index('Date', inplace=True)
        
        return gpr_long