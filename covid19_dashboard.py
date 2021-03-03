import pandas as pd, numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import streamlit as st
import streamlit.components.v1 as components

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
    def make_df2():
        test = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv')
        test.rename(columns={'iso_code':'country_code_3', 'location':'country', 'date':'date_parsed', 'total_cases':'cases', 'total_deaths':'deaths'}, inplace=True)
        #test = test[test.country != 'World']
        test.sort_values(by='date_parsed', inplace=True)
        num = test._get_numeric_data()
        num[num < 0] = 0
        #test = pd.melt(test, id_vars=['country_code_3', 'country', 'date_parsed'], value_vars=['total_cases', 'total_deaths', 'total_cases_per_million', 'total_deaths_per_million'], var_name='category', value_name='cases')
        return test
    
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

    def make_map(current, variable):
      fig = px.choropleth(current, locations="country_code_3",
                    color=variable,
                    hover_name="country",
                    color_continuous_scale=px.colors.diverging.Spectral_r
                    )
      fig.update_layout(height=500,coloraxis_colorbar=dict(title=""))
      return fig

    def make_map(df, variable):
            fig = go.Figure(data=go.Choropleth(
            locations=df['country_code_3'],
            z=df[variable].apply(lambda x: np.round(x,2)),
            locationmode='ISO-3',
            #colorscale='Reds',
            colorscale=px.colors.diverging.Spectral_r,
            autocolorscale=False,
            text=df["country"], # hover text
            marker_line_color='black', # line markers between states
            colorbar_title="",
            #hovertemplate='%{z:.2f}'+'<br>%{locations}<br>',
            marker_line_width=0.5
            )
            )
            fig.update_layout(
            title_text='',
            geo = dict(

            #projection=go.layout.geo.Projection(type = 'albers usa'),
            #showlakes=True, # lakes
            #lakecolor='rgb(255, 255, 255)'),
            ))
            return fig

    def plot_ts(df, x, y, countries):
        df = df[df.days_since_n > 0].loc[countries,:].reset_index()
        fig = go.Figure()
        for country in countries:
            dfc = df[df['country'] == country]
            fig.add_trace(go.Scatter(x=dfc[x], y=dfc[y],
                            mode='lines',
                            name=country,
                            hovertemplate='%{y:.2f}'+
                            '<br>%{x}<br>',        
                                    ))
        fig.update_layout(xaxis_title="", yaxis_title="", height=500, template="plotly_white",
        #legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01)
            legend={"orientation":'h'}
            )
        return fig

    @st.cache(ttl=60*60*3)
    def build_df():
        df = make_df2()
        output = add_days_since_n(df, n=100)
        df = output.set_index(['country', 'date_parsed']).copy()
        df.rename(columns={'total_deaths_per_million':'deaths_per_million', 'total_cases_per_million':'cases_per_million'}, inplace=True)
        df.sort_values(by=['country', 'date_parsed'],inplace=True)
        df = compute_new_cases(df, window=7)
        df = compute_new_cases_pop(df, window=7)
        df.reset_index('date_parsed', drop=False, inplace=True)
        return df

    ask_refresh = False
    st.title("Covid-19 Dashboard")
    df = build_df()
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    
    def update_current(df, chosen_date):
        current = df[df.date_parsed==chosen_date].reset_index()
        current = current[current['population'] > 10**6]
        return current

    countries = list(np.unique(df.index.values))
    st.sidebar.text(f'Last update: {df.date_parsed.max()}')
    if df.date_parsed.max() != datetime.now().strftime('%Y-%m-%d'):
      ask_refresh = st.sidebar.button("Refresh data")
    
    choice_category = st.sidebar.radio("Category:",('Cases', 'Deaths', 'Reproduction rate', 'Positive rate'), index = 0)
    category = str.lower(choice_category)
    if choice_category in ['Cases', 'Deaths']:
        choice_variable = st.sidebar.radio("Evolution:",('Cumulative', 'Daily'), index = 1)
        variable = "_speed" if choice_variable == "Daily" else ""
        choice_smoothed = st.sidebar.radio("Weekly rolling average",('Yes', 'No'), index = 0)
        smoothed = "_smoothed" if choice_smoothed == "Yes" else ""
        text_smoothed = ', weekly rolling average' if choice_smoothed == "Yes" else ""    
        choice_perm = st.sidebar.radio("Normalize by population:",('Yes', 'No'), index = 0)
        perm = "_per_million" if choice_perm == "Yes" else ""
        text_perm = ' per million inhabitants' if choice_perm == "Yes" else ""

    if choice_category in ['Cases', 'Deaths']:
        y = category+perm+variable+smoothed
        plot_title = f'{choice_variable} {category}{text_perm}{text_smoothed}'
    if choice_category == 'Reproduction rate':
        y = 'reproduction_rate'
        plot_title = choice_category
    if choice_category == 'Positive rate':
        y = 'positive_rate'
        plot_title = choice_category
    if ask_refresh:
      if df.date_parsed.max() != datetime.now().strftime('%Y-%m-%d'):
        df = build_df()
    st.subheader(plot_title)
    chosen_date = str(st.date_input("Map date:", yesterday))
    st.write(make_map(update_current(df, chosen_date), y))
    #st.write(myplotly(df, 'date_parsed', y, choice_countries, "cases"))
    choice_countries = st.multiselect('Choose countries:', countries, 
                                              default = ['France', 'Spain', "United Kingdom"])
    st.plotly_chart(plot_ts(df, 'date_parsed', y, choice_countries))
    st.sidebar.write("Source data can be found [here](https://github.com/owid/covid-19-data/tree/master/public/data)")

if __name__ == '__main__':
    main()
