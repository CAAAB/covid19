import pandas as pd, numpy as np
from scipy.stats import uniform, bernoulli
from random import choices, sample
from itertools import combinations
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
#import pygraphviz as pgv
import streamlit as st
import streamlit.components.v1 as components

PAGE_CONFIG = {"page_title":"Covid-19 simulation","page_icon":":mask:","layout":"wide"}
st.set_page_config(**PAGE_CONFIG)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def main():
    def random_combination(iterable, r):
        "Random selection from itertools.combinations(iterable, r)"
        pool = tuple(iterable)
        n = len(pool)
        indices = sorted(sample(range(n), r))
        return tuple(pool[i] for i in indices)

    class Community:
        def __init__(self, size, p_susceptible, p_infected, p_recovered, disease, vaccination_strategy):
            self.p_susceptible = p_susceptible
            self.p_infected = p_infected
            self.p_recovered = p_recovered
            self.susceptible = []
            self.infected = []
            self.recovered = []
            self.dead = []
            self.vaccinated = []
            self.size = size
            self.time = 0
            self.meeting_log = []
            self.vaccination_strategy = vaccination_strategy
            statuses = choices(['susceptible', 'infected', 'recovered'], [p_susceptible, p_infected, p_recovered], k=size)
            activities = np.random.uniform(size=size)
            fragilities = np.random.uniform(size=size)
            self.population = [Person(i, statuses[i], activity=activities[i], disease=disease, fragility=fragilities[i]) for i in range(size)]
            self.people_vaccinated = self.vaccinate()
            self.update_community_stats()

        def update_community_stats(self):
            statuses = dict(self.population_df().status.value_counts())
            dead = statuses['dead'] if "dead" in statuses.keys() else 0
            self.dead.append(dead)
            susceptible = statuses['susceptible'] if "susceptible" in statuses.keys() else 0
            self.susceptible.append(susceptible)
            infected = statuses['infected'] if "infected" in statuses.keys() else 0
            self.infected.append(infected)
            recovered = statuses['recovered'] if "recovered" in statuses.keys() else 0
            self.recovered.append(recovered)
            vaccinated = statuses['vaccinated'] if "vaccinated" in statuses.keys() else 0
            self.vaccinated.append(vaccinated)
            self.size = len(self.population)

        def evolve(self):
            active_people = np.where([bernoulli.rvs(person.activity) for person in self.population])[0]
            #possible_meetings = list(combinations(active_people,2))
            #meetings = choices(possible_meetings,k=len(possible_meetings)//2) # Half of the possible meetings actually occur
            meetings = [random_combination(active_people,2) for _ in range(self.size//5)]
            for i, j in meetings:
                new_entry = self.population[i].meet(self.population[j], self.time)
                self.meeting_log.append(new_entry[0])
                self.meeting_log.append(new_entry[1])
            for person in self.population:
                person.step()
            self.time += 1
            self.update_community_stats()
            return 'infected' in self.population_status().keys()

        def vaccinate(self):
            number_vaccinated = 0
            if self.vaccination_strategy == "fragile":
                for person in self.population:
                    if person.fragility >= .91:
                        person.status='vaccinated'
                        number_vaccinated += 1
            elif self.vaccination_strategy == "active":
                for person in self.population:
                    if person.activity >= .91:
                        person.status="vaccinated"
                        number_vaccinated += 1
            return number_vaccinated

        def add_people(self, people):
            self.population.append(people)
            self.update_community_stats()

        def print(self):
            statuses = [person.get_status() for person in self.population]
            print(pd.DataFrame(pd.Series(statuses).value_counts()))

        def population_df(self):
            return pd.DataFrame([person.get_record() for person in self.population])

        def population_status(self):
            vals = [person.get_status() for person in self.population]
            unique_vals = np.unique(vals)
            population_status = {'time':self.time}
            for status in unique_vals:
                population_status[status]=vals.count(status)
            return population_status


        def render_community_graph(self):
            meeting_log = pd.DataFrame(self.meeting_log)
            G = nx.from_pandas_edgelist(meeting_log[meeting_log.gave_virus == True], "id", 'id_contact')
            #pos = nx.spring_layout(G)
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot") # dot, neato, twopi, circo
            d = dict(G.degree)
            Xv=[pos[k][0] for k in G.nodes]
            Yv=[pos[k][1] for k in G.nodes]
            Xed=[]
            Yed=[]
            for edge in G.edges:
                Xed+=[pos[edge[0]][0],pos[edge[1]][0], None]
                Yed+=[pos[edge[0]][1],pos[edge[1]][1], None]

            trace3=go.Scatter(x=Xed,
                          y=Yed,
                          mode='lines',
                          line=dict(color='rgb(210,210,210)', width=1),
                          hoverinfo='none'
                          )
            trace4=go.Scatter(x=Xv,
                          y=Yv,
                          mode='markers',
                          name='Person',
                          #color=d.values(),
                          marker=dict(symbol='circle-dot',
                                        size=5,
                                        color='DarkSlateGrey',
                                        line=dict(color='rgb(50,50,50)', width=0.5)
                                        ),
                          text=[n for n in G.nodes],
                          hoverinfo='text'
                          )

            fig=go.Figure([trace3, trace4])
            fig.update_layout(template='simple_white', xaxis= {
                'showgrid': False, # thin lines in the background
                'zeroline': False, # thick line at x=0
                'visible': False,  # numbers below
            },yaxis= {
                'showgrid': False, # thin lines in the background
                'zeroline': False, # thick line at x=0
                'visible': False,  # numbers below
            },
            showlegend=False
            )
            return fig

    class Disease:
        def __init__(self, infectiousness, recovery_time, mortality):
            self.infectiousness = infectiousness
            self.recovery_time = recovery_time
            self.mortality = mortality


    class Person:
        def __init__(self, id, status, activity, disease, fragility):
            self.id=id
            self.status=status
            self.activity=activity
            self.history=[]
            self.disease=disease
            self.days_infected=0
            self.contact_log=[]
            self.fragility = fragility

        def get_status(self):
            return self.status

        def set_status(self, new_status):
            self.status=new_status

        def get_record(self):
            return {"id":self.id, "status":self.status, "activity":self.activity, 'fragility':self.fragility, "days_infected":self.days_infected}

        def get_infected(self):
            infection_result = False
            if self.status == "susceptible":
                self.status='infected'
                infection_result = True
            return infection_result

        def step(self):
            if self.status == 'infected':
                self.days_infected +=1
                if self.days_infected >= self.disease.recovery_time:
                    if bernoulli.rvs(self.disease.mortality*(1+self.fragility)):
                        self.status='dead'
                    else:
                        self.status='recovered'
            self.history.append(self.status)

        def meet(self, other, time):
            gave_virus = False
            caught_virus = False
            if self.status == 'infected':
                if bernoulli.rvs(self.disease.infectiousness):
                    gave_virus = other.get_infected()    
            elif other.get_status() == 'infected':
                if bernoulli.rvs(self.disease.infectiousness):
                    caught_virus = self.get_infected()
            my_log = {'time':time, 'id':self.id, 'id_contact':other.id, 'gave_virus':gave_virus, 'caught_virus':caught_virus}
            their_log = {'time':time,'id':other.id, 'id_contact':self.id, 'gave_virus':caught_virus, 'caught_virus':gave_virus}
            self.contact_log.append(my_log)
            other.contact_log.append(their_log)
            return [my_log, their_log]

    disease = Disease(infectiousness=.75, recovery_time=14, mortality=.2)
    com = Community(size=500, p_susceptible=1, p_infected=0, p_recovered=0, disease=disease, vaccination_strategy="fragile")
    com.add_people(Person(len(com.population),'infected', .99, disease, .5))
    still_some_infected = True
    while com.time < 100 and still_some_infected:
        still_some_infected = com.evolve()
    df = pd.DataFrame({'susceptible':com.susceptible, 'infected':com.infected, 'recovered':com.recovered, 'dead':com.dead, 'vaccinated':com.vaccinated})
    st.write(plt.plot(df[1:]))
    st.write(pd.DataFrame([com.population_status()]).drop('time', axis=1).plot.bar())
    com.population_status()
    st.write(com.render_community_graph())
    
    if __name__ == '__main__':
        main()
