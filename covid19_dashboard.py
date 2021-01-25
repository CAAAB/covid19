import pandas as pd, numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import streamlit as st

PAGE_CONFIG = {"page_title":"Covid-19 dashboard","page_icon":":mask:","layout":"wide"}
st.set_page_config(**PAGE_CONFIG)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
def main():
    premade = True
    def make_df2():
        test = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv')
        test.rename(columns={'iso_code':'country_code_3', 'location':'country', 'date':'date_parsed', 'total_cases':'cases', 'total_deaths':'deaths'}, inplace=True)
        #test = test[test.country != 'World']
        test.sort_values(by='date_parsed', inplace=True)
        num = test._get_numeric_data()
        num[num < 0] = 0
        #test = pd.melt(test, id_vars=['country_code_3', 'country', 'date_parsed'], value_vars=['total_cases', 'total_deaths', 'total_cases_per_million', 'total_deaths_per_million'], var_name='category', value_name='cases')
        return test
    
    def make_df_pop():
        df_pop = pd.read_csv('https://raw.githubusercontent.com/datasets/population/master/data/population.csv')
        df_pop = df_pop[df_pop.Year == df_pop.Year.max()]
        df_pop = df_pop.rename(columns={'Country Name':'country', 'Country Code':'country_code_3', 'Value':'Population'}).drop(columns='Year')
        return df_pop
        
    def make_df():
        url_start = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_'
        df = None
        for category in ['confirmed', 'deaths', 'recovered']:
            dd = pd.read_csv(url_start + category + '_global.csv')
            dd['category'] = category
            df = pd.concat([df, dd]) if df is not None else dd
            
        df = df[['category']+df.columns.values[:-1].tolist()]
        df = pd.melt(df,id_vars=['category','Province/State', 'Country/Region', 'Lat', 'Long'],
                                value_vars=df.columns.values[5:],
                                var_name='date',value_name='cases')
        
        latlong = list(df[['Lat', 'Long']].itertuples(index=False, name=None))
        
        countrylist = rvg.search(latlong)
        
        df['country_code_2'] = [x['cc'] for x in countrylist]
        df['country_code_3'] = df['country_code_2']
        
        convert_country_codes = {'AF':'AFG','AX':'ALA','AL':'ALB','DZ':'DZA','AS':'ASM','AD':'AND','AO':'AGO','AI':'AIA','AQ':'ATA','AG':'ATG','AR':'ARG','AM':'ARM',
                                'AW':'ABW','AU':'AUS','AT':'AUT','AZ':'AZE','BS':'BHS','BH':'BHR','BD':'BGD','BB':'BRB','BY':'BLR','BE':'BEL','BZ':'BLZ','BJ':'BEN',
                                'BM':'BMU','BT':'BTN','BO':'BOL','BA':'BIH','BW':'BWA','BV':'BVT','BR':'BRA','IO':'IOT','BN':'BRN','BG':'BGR','BF':'BFA','BI':'BDI',
                                'KH':'KHM','CM':'CMR','CA':'CAN','CV':'CPV','KY':'CYM','CF':'CAF','TD':'TCD','CL':'CHL','CN':'CHN','CX':'CXR','CC':'CCK','CO':'COL',
                                'KM':'COM','CG':'COG','CD':'COD','CK':'COK','CR':'CRI','CI':'CIV','HR':'HRV','CU':'CUB','CY':'CYP','CZ':'CZE','DK':'DNK','DJ':'DJI',
                                'DM':'DMA','DO':'DOM','EC':'ECU','EG':'EGY','SV':'SLV','GQ':'GNQ','ER':'ERI','EE':'EST','ET':'ETH','FK':'FLK','FO':'FRO','FJ':'FJI',
                                'FI':'FIN','FR':'FRA','GF':'GUF','PF':'PYF','TF':'ATF','GA':'GAB','GM':'GMB','GE':'GEO','DE':'DEU','GH':'GHA','GI':'GIB','GR':'GRC',
                                'GL':'GRL','GD':'GRD','GP':'GLP','GU':'GUM','GT':'GTM','GG':'GGY','GN':'GIN','GW':'GNB','GY':'GUY','HT':'HTI','HM':'HMD','VA':'VAT',
                                'HN':'HND','HK':'HKG','HU':'HUN','IS':'ISL','IN':'IND','ID':'IDN','IR':'IRN','IQ':'IRQ','IE':'IRL','IM':'IMN','IL':'ISR','IT':'ITA',
                                'JM':'JAM','JP':'JPN','JE':'JEY','JO':'JOR','KZ':'KAZ','KE':'KEN','KI':'KIR','KP':'PRK','KR':'KOR','KW':'KWT','KG':'KGZ','LA':'LAO',
                                'LV':'LVA','LB':'LBN','LS':'LSO','LR':'LBR','LY':'LBY','LI':'LIE','LT':'LTU','LU':'LUX','MO':'MAC','MK':'MKD','MG':'MDG','MW':'MWI',
                                'MY':'MYS','MV':'MDV','ML':'MLI','MT':'MLT','MH':'MHL','MQ':'MTQ','MR':'MRT','MU':'MUS','YT':'MYT','MX':'MEX','FM':'FSM','MD':'MDA',
                                'MC':'MCO','MN':'MNG','ME':'MNE','MS':'MSR','MA':'MAR','MZ':'MOZ','MM':'MMR','NA':'NAM','NR':'NRU','NP':'NPL','NL':'NLD','AN':'ANT',
                                'NC':'NCL','NZ':'NZL','NI':'NIC','NE':'NER','NG':'NGA','NU':'NIU','NF':'NFK','MP':'MNP','NO':'NOR','OM':'OMN','PK':'PAK','PW':'PLW',
                                'PS':'PSE','PA':'PAN','PG':'PNG','PY':'PRY','PE':'PER','PH':'PHL','PN':'PCN','PL':'POL','PT':'PRT','PR':'PRI','QA':'QAT','RE':'REU',
                                'RO':'ROU','RU':'RUS','RW':'RWA','BL':'BLM','SH':'SHN','KN':'KNA','LC':'LCA','MF':'MAF','PM':'SPM','VC':'VCT','WS':'WSM','SM':'SMR',
                                'ST':'STP','SA':'SAU','SN':'SEN','RS':'SRB','SC':'SYC','SL':'SLE','SG':'SGP','SK':'SVK','SI':'SVN','SB':'SLB','SO':'SOM','ZA':'ZAF',
                                'GS':'SGS','ES':'ESP','LK':'LKA','SD':'SDN','SR':'SUR','SJ':'SJM','SZ':'SWZ','SE':'SWE','CH':'CHE','SY':'SYR','TW':'TWN','TJ':'TJK',
                                'TZ':'TZA','TH':'THA','TL':'TLS','TG':'TGO','TK':'TKL','TO':'TON','TT':'TTO','TN':'TUN','TR':'TUR','TM':'TKM','TC':'TCA','TV':'TUV',
                                'UG':'UGA','UA':'UKR','AE':'ARE','GB':'GBR','US':'USA','UM':'UMI','UY':'URY','UZ':'UZB','VU':'VUT','VE':'VEN','VN':'VNM','VG':'VGB',
                                'VI':'VIR','WF':'WLF','EH':'ESH','YE':'YEM','ZM':'ZMB','ZW':'ZWE'}
        
        df.replace({"country_code_3":convert_country_codes},inplace=True)
        df['date_parsed'] = pd.to_datetime(df['date'], format='%m/%d/%y')
        
        # Add world population
        #df_pop = get_dataset("world_pop_2018")[['Country Name', 'Country Code', '2018']]
        #df_pop.rename(columns={'Country Name':'country', 'Country Code':'country_code_3', '2018':'Population'}, inplace=True)
        df = df.merge(make_df_pop(), left_on='country_code_3', right_on='country_code_3')
        
        # Normalize by world population
        df['cases_pop'] = 1000000*df['cases']/df['Population']
        return df
    
    
    def add_days_since_n(df, n):
        out = None
        for state in df['country'].unique():
            #print(state)
            co = df[df['country'] == state].copy()
            italy_confirmed = co[co['category'] == 'confirmed'].copy()
            tenormore = italy_confirmed['date_parsed'][italy_confirmed['cases']>n].copy()
            if len(tenormore) > 0:
                date_threshold = tenormore.values[0]
                italy_confirmed.loc[:,'days_since_n'] = (1*(italy_confirmed['date_parsed'] >= date_threshold)).cumsum().values
                italy_confirmed = italy_confirmed.loc[:,['date_parsed', 'days_since_n']].copy()
                italy_confirmed.loc[:, 'country'] = state
                out = italy_confirmed if out is None else pd.concat([out, italy_confirmed])
        output = pd.merge(df, out, how='left')
        return output
    
    def add_days_since_n(df, n):
        out = None
        for state in df['country'].unique():
            #print(state)
            co = df[df['country'] == state].copy()
            italy_confirmed = co
            tenormore = italy_confirmed['date_parsed'][italy_confirmed['cases']>n].copy()
            if len(tenormore) > 0:
                date_threshold = tenormore.values[0]
                italy_confirmed.loc[:,'days_since_n'] = (1*(italy_confirmed['date_parsed'] >= date_threshold)).cumsum().values
                italy_confirmed = italy_confirmed.loc[:,['date_parsed', 'days_since_n']].copy()
                italy_confirmed.loc[:, 'country'] = state
                out = italy_confirmed if out is None else pd.concat([out, italy_confirmed])
        output = pd.merge(df, out, how='left')
        return output
    
    def my_smoothie(df, x, window):
        return df[x].rolling(window).mean().ewm(span=3).mean()
    
    def compute_new_cases(tdf, window):
        for category in ['cases', 'deaths']:
            tdf[category+'_smoothed'] = my_smoothie(tdf, category, window)
            tdf[category+'_speed'] = tdf[category+''].groupby(['country']).diff()
            tdf[category+'_speed_smoothed'] = my_smoothie(tdf,category+'_speed', window)
            tdf[category+'_acceleration'] = tdf[category+'_speed_smoothed'].diff()
            tdf[category+'_acceleration_smoothed'] = my_smoothie(tdf,category+'_acceleration', window)
        return tdf
    
    def compute_new_cases_pop(tdf, window):
        for category in ['cases', 'deaths']:
            tdf[category+'_per_million_smoothed'] = my_smoothie(tdf, category+'_per_million', window)
            tdf[category+'_per_million_speed'] = tdf[category+'_per_million'].groupby(['country']).diff()
            tdf[category+'_per_million_speed_smoothed'] = my_smoothie(tdf,category+'_per_million_speed', window)
            tdf[category+'_per_million_acceleration'] = tdf[category+'_per_million_speed_smoothed'].diff()
            tdf[category+'_per_million_acceleration_smoothed'] = my_smoothie(tdf,category+'_per_million_acceleration', window)
        return tdf
    
    def myplot(df, x, y, countries, category, scatter=False):
        fig, ax = plt.subplots(figsize=(15,12))
        for country, cdf in df[df.days_since_n > 0].loc[(countries, category),:].groupby('country'):
            if scatter:
                cdf.plot(kind='scatter',x='days_since_n',y=y, ax=ax, label=country)
            cdf.plot(kind='line',x=x,y=y, ax=ax, label=country, linewidth=3)
        ax.axhline(y=0, color='black')
        grid(b=True, which='major', color='lightgray', linestyle='-', axis='y')
        ax.tick_params(axis='both', labelsize=16)
        plt.show()
    
    def myplotly(df, x, y, countries, category, scatter = False):
        fig = px.line(df[df.days_since_n > 0].loc[countries,:].reset_index(), x=x, y=y, title="", color='country', template='plotly_white').for_each_trace(lambda t: t.update(name=t.name.split("=")[1]))
        fig.update_layout(xaxis_title="", yaxis_title="", height=500,
        #legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01)
        legend={"orientation":'h'}
        )

        return fig

    def make_map(df, variable):
      yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
      #chosen_date = df.date_parsed.max()
      chosen_date = yesterday
      current = df[df.date_parsed==chosen_date].reset_index()
      current = current[current['population'] > 10**6]
      fig = px.choropleth(current, locations="country_code_3",
                    color=variable,
                    hover_name="country",
                    color_continuous_scale=px.colors.diverging.Spectral_r
                    )
      fig.update_layout(height=500,coloraxis_colorbar=dict(title=""))
      return fig

    def plot_date(df, y , countries, category, scatter=False, smooth=False):
        fig, ax = plt.subplots(figsize=(10,10))
        smooth = '_smoothed' if smooth else ''
        for country, cdf in df[df.days_since_n > 0].loc[(countries, category),:].groupby('country'):
            if scatter:
                cdf.plot(kind='scatter',x='days_since_n',y=y, ax=ax, label=country)
            cdf.plot(kind='line',x='date_parsed',y=y+smooth, ax=ax, label=country, linewidth=3)
        ax.axhline(y=0, color='black')
        grid(b=True, which='major', color='lightgray', linestyle='-', axis='y')
        ax.tick_params(axis='both', labelsize=16)
        plt.show()

    def make_france():
        df = pd.read_csv('https://raw.githubusercontent.com/opencovid19-fr/data/master/dist/chiffres-cles.csv')
        dep = df[df.granularite == "departement"]
        dep['maille_code'] = dep.maille_code.str.split("-",expand=True).loc[:,1]
        dep.rename(columns={'maille_code':'dep_id'}, inplace=True)

        dep_pop = pd.DataFrame(pd.read_html("https://en.wikipedia.org/wiki/List_of_French_departments_by_population")[0])[["Legal population in 2013", "INSEE Dept. No."]]
        dep_pop.rename(columns={"Legal population in 2013":"population", "INSEE Dept. No.":"dep_id"}, inplace=True)
        dep_pop

        return pd.merge(dep,dep_pop)

    def get_indices(olist, clist):
        default_ix = []
        for it in clist:
            default_ix.append(olist.index(it))
        return default_ix

    @st.cache(persist=True)
    def build_df():
        df = make_df2() if premade else make_df()
        output = add_days_since_n(df, n=100)
        df = output.set_index(['country', 'date_parsed']).copy()
        df.rename(columns={'total_deaths_per_million':'deaths_per_million', 'total_cases_per_million':'cases_per_million'}, inplace=True)
        df.sort_values(by=['country', 'date_parsed'],inplace=True)
        df = compute_new_cases(df, window=7)
        df = compute_new_cases_pop(df, window=7)
        df.reset_index('date_parsed', drop=False, inplace=True)
        return df

    st.title("Covid-19 Dashboard")
    df = build_df()
    countries = list(np.unique(df.index.values))
    st.sidebar.text(f'Last update: {df.date_parsed.max()}')
    if df.date_parsed.max() != datetime.now().strftime('%Y-%m-%d'):
      ask_refresh = st.sidebar.button("Refresh data")
    
    choice_category = st.sidebar.radio("Category:",('Cases', 'Deaths'), index = 0)
    category = str.lower(choice_category)
    choice_variable = st.sidebar.radio("Evolution:",('Cumulative', 'Daily'), index = 1)
    variable = "_speed" if choice_variable == "Daily" else ""
    
    choice_perm = st.sidebar.radio("Normalize by population:",('Yes', 'No'), index = 0)
    perm = "_per_million" if choice_perm == "Yes" else ""
    text_perm = ' per million inhabitants' if choice_perm == "Yes" else ""
    choice_smoothed = st.sidebar.radio("Weekly rolling average",('Yes', 'No'), index = 0)
    smoothed = "_smoothed" if choice_smoothed == "Yes" else ""
    text_smoothed = ', weekly rolling average' if choice_smoothed == "Yes" else ""
    y = category+perm+variable+smoothed

    if ask_refresh:
      if df.date_parsed.max() != datetime.now().strftime('%Y-%m-%d'):
        df = build_df()
    st.subheader(f'Latest {str.lower(choice_variable)} {category}{text_perm}{text_smoothed}:')
    st.write(make_map(df, y))
    st.subheader(f'{choice_variable} {category}{text_perm}{text_smoothed}')
    #st.write(myplotly(df, 'date_parsed', y, choice_countries, "cases"))
    choice_countries = st.multiselect('Choose countries:', countries, 
                                              default = ['France', 'Spain', "United Kingdom"])
    st.plotly_chart(myplotly(df, 'date_parsed', y, choice_countries, "cases"))
    link = f'[Source: Raw data can be found here]("https://covid.ourworldindata.org/data/owid-covid-data.csv")'
    st.markdown(link, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
