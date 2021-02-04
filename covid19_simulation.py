import pandas as pd, numpy as np
from scipy.stats import uniform, bernoulli, beta
from random import choices, sample, seed
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

    def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

        '''
        From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
        Licensed under Creative Commons Attribution-Share Alike 
        
        If the graph is a tree this will return the positions to plot this in a 
        hierarchical layout.
        
        G: the graph (must be a tree)
        
        root: the root node of current branch 
        - if the tree is directed and this is not given, 
          the root will be found and used
        - if the tree is directed and this is given, then 
          the positions will be just for the descendants of this node.
        - if the tree is undirected and not given, 
          then a random choice will be used.
        
        width: horizontal space allocated for this branch - avoids overlap with other branches
        
        vert_gap: gap between levels of hierarchy
        
        vert_loc: vertical location of root
        
        xcenter: horizontal location of root
        '''
        if not nx.is_tree(G):
            raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

        if root is None:
            if isinstance(G, nx.DiGraph):
                root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
            else:
                root = random.choice(list(G.nodes))

        def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
            '''
            see hierarchy_pos docstring for most arguments

            pos: a dict saying where all nodes go if they have been assigned
            parent: parent of this branch. - only affects it if non-directed

            '''
        
            if pos is None:
                pos = {root:(xcenter,vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            children = list(G.neighbors(root))
            if not isinstance(G, nx.DiGraph) and parent is not None:
                children.remove(parent)  
            if len(children)!=0:
                dx = width/len(children) 
                nextx = xcenter - width/2 - dx/2
                for child in children:
                    nextx += dx
                    pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                        vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                        pos=pos, parent = root)
            return pos
            
        return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
        
    class Community:
        def __init__(self, size, vaccination_strategy):
            self.size = size
            self.lockdown_factor = 1
            self.evolution = None
            self.time = 0
            self.meeting_log = []
            self.vaccination_strategy = vaccination_strategy
            activities = np.random.uniform(size=size)
            #fragilities = np.random.uniform(size=size)
            fragilities = 1-activities
            self.population = [Person(i, "susceptible", activity=activities[i], disease="none", fragility=fragilities[i]) for i in range(size)]
            self.people_vaccinated = self.vaccinate()
            self.update_community_stats()
            self.time += 1
            #self.evolution = self.current_pop_status()
      
        def initiate_lockdown(self, lockdown_factor):
            self.lockdown_factor = lockdown_factor

        def lift_lockdown(self):
            self.lockdown_factor = 1

        def R0(self):
            df = pd.DataFrame(self.meeting_log)
            return df[df['gave_virus'] == True].groupby('id').agg('sum').reset_index()['gave_virus'].mean()

        def current_pop_status(self):
            current_pop_status = {f'{status}':[np.sum(self.population_df()['status'] == status)] for status in ['susceptible', 'infected', 'recovered', 'dead', 'vaccinated']}
            current_pop_status['time'] = [self.time]
            return pd.DataFrame(current_pop_status)
      
        def update_community_stats(self):
            self.evolution = pd.concat([self.evolution, self.current_pop_status()], ignore_index=False) if self.evolution is not None else self.current_pop_status() 
          
        def evolve(self):
            # improvement: only infected and susceptible actually participate here
            #i_or_s = np.where([x in ['infected', 'susceptible'] for x in np.array(self.population_df()['status'])])[0]
            #active_people = np.where([bernoulli.rvs(person.activity*self.lockdown_factor) for person in np.array(self.population)[i_or_s]])[0]
            active_people = np.where([bernoulli.rvs(person.activity*self.lockdown_factor) for person in self.population])[0]
            if len(active_people) >=2:
                #possible_meetings = list(combinations(active_people,2))
                #meetings = choices(possible_meetings,k=len(possible_meetings)//2) # Half of the possible meetings actually occur
                meetings = list(set([random_combination(active_people,2) for _ in range(self.size//5)]))
                for i, j in meetings:
                    new_entry = self.population[i].meet(self.population[j], self.time)
                    self.meeting_log.append(new_entry[0])
                    self.meeting_log.append(new_entry[1])
            for person in self.population:
                person.step()
            self.update_community_stats()
            self.time += 1
            return self.current_pop_status()['infected'][0] != 0
        
        def plot_evolution(self):
            fig = go.Figure()
            for status in statuses:
                fig.add_trace(go.Scatter(x=self.evolution.time, y=self.evolution[status],
                                mode='lines',
                                name=status,
                                #hovertemplate='%{x}'+'<br>%{y}<br>'
                                ))
            fig.update_layout(xaxis_title="Time", yaxis_title="Count", height=500, template="plotly_white",
            #legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01)
                #legend={"orientation":'h'}
                )
            plt.show()
            return fig

        def vaccinate(self):
            number_vaccinated = 0
            if self.vaccination_strategy == "fragile":
                for person in self.population:
                    if person.fragility >= .9:
                        person.status='vaccinated'
                        number_vaccinated += 1
            elif self.vaccination_strategy == "active":
                for person in self.population:
                    if person.activity >= .9:
                        person.status="vaccinated"
                        number_vaccinated += 1
            return number_vaccinated
      
        def add_people(self, person):
            self.population.append(person)
        
        def print(self):
            statuses = [person.get_status() for person in self.population]
            print(pd.DataFrame(pd.Series(statuses).value_counts()))
      
        def population_df(self):
            return pd.DataFrame([person.get_record() for person in self.population])
      
        def population_status(self):
            vals = [person.status for person in self.population]
            unique_vals = np.unique(vals)
            population_status = {'time':self.time}
            for status in unique_vals:
                population_status[status]=vals.count(status)
            return population_status
      
        def render_community_graph(self, layout="dot"):
            meeting_log = pd.DataFrame(com.meeting_log)
            if meeting_log.shape[0] < 2:
                return None
            
            G = nx.from_pandas_edgelist(meeting_log[meeting_log.gave_virus == True], "id", 'id_contact', create_using=nx.DiGraph())
            #pos = nx.spring_layout(G)
            try:
                pos = hierarchy_pos(G)   
            except TypeError:
                pos = nx.spring_layout(G)
            #pos = nx.nx_agraph.graphviz_layout(G, prog=layout) # dot, neato, twopi, circo
            d = dict(G.degree)
            Xv=[pos[k][0] for k in G.nodes]
            Yv=[pos[k][1] for k in G.nodes]
            Xed=[]
            Yed=[]
            for edge in G.edges:
                Xed+=[pos[edge[0]][0],pos[edge[1]][0], None]
                Yed+=[pos[edge[0]][1],pos[edge[1]][1], None]
        
            trace3=go.Scatter(x=Xed,y=Yed,mode='lines',line=dict(color='rgb(210,210,210)', width=1),hoverinfo='none')
            trace4=go.Scatter(x=Xv, y=Yv,mode='markers',name='Person',marker=dict(symbol='circle-dot',
                                        size=5,
                                        color='DarkSlateGrey',
                                        line=dict(color='rgb(50,50,50)', width=0.5)
                                        ),
                            text=[n for n in G.nodes],hoverinfo='text'
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
        def __init__(self, name, infectiousness, recovery_time, mortality):
            self.name = name
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
      
        def get_record(self):
            return {"id":self.id, "status":self.status, "activity":self.activity, 'fragility':self.fragility, "days_infected":self.days_infected}
      
        def get_infected(self, disease):
            infection_result = False
            if self.status == "susceptible":
                self.status = 'infected'
                self.disease = disease
                infection_result = True
            return infection_result
      
        def step(self):
            if self.status == 'infected':
                self.days_infected +=1
                if self.days_infected >= self.disease.recovery_time:
                    if bernoulli.rvs(min([1,self.disease.mortality*(1+1-np.tanh(13-self.fragility*13))])):
                        self.status='dead'
                    else:
                        self.status='recovered'
            self.history.append(self.status)
      
        def meet(self, other, time):
            gave_virus = False
            caught_virus = False
            if self.status == 'infected' and other.status == 'susceptible':
                if bernoulli.rvs(self.disease.infectiousness):
                    gave_virus = other.get_infected(self.disease)    
            elif other.status == 'infected' and self.status == "susceptible":
                if bernoulli.rvs(other.disease.infectiousness):
                    caught_virus = self.get_infected(other.disease)
            my_log = {'time':time, 'id':self.id, 'id_contact':other.id, 'gave_virus':gave_virus, 'caught_virus':caught_virus}
            their_log = {'time':time,'id':other.id, 'id_contact':self.id, 'gave_virus':caught_virus, 'caught_virus':gave_virus}
            self.contact_log.append(my_log)
            other.contact_log.append(their_log)
            return [my_log, their_log]

    ui_community_size = st.sidebar.slider("Population", 10,2010, 510, 100)
    st.sidebar.subheader("Strategy")
    #ui_lockdown = st.sidebar.radio("Lockdown",("Enabled", "Disabled"), index=1)
    ui_vaccination_strategy = st.sidebar.radio("Vaccinate",("None", "Fragile", "Active"), index=0)
    st.sidebar.subheader("Disease")
    ui_infectiousness = st.sidebar.slider("Infectiousness", 0,100,70,1)
    ui_recovery_time = st.sidebar.slider("Recovery time", 1,90,7,1)
    ui_mortality = st.sidebar.slider("Mortality", 0,100,20,1)
    
    #lockdown = True if ui_lockdown == "Enabled" else False
    vaccination_strategy = str.lower(ui_vaccination_strategy)
    size = ui_community_size
    infectiousness = ui_infectiousness/100
    recovery_time = ui_recovery_time
    mortality = ui_mortality/100
    
    statuses = ['susceptible', 'infected', 'recovered', 'dead', 'vaccinated']
    graine = 42
    
    @st.cache(allow_output_mutation=True)
    def make_community(size, vaccination_strategy, graine):
        seed(graine)
        return Community(size=size, vaccination_strategy=vaccination_strategy)
        
    com = make_community(size=size, vaccination_strategy=vaccination_strategy, graine=graine)
    disease = Disease(name="disease", infectiousness=infectiousness, recovery_time=recovery_time, mortality=mortality)
    
    graine = st.number_input("Seed", 1, 9999, graine, 1)
    
    press_reset_community = st.button("Reset community")
    if press_reset_community:
        com = make_community(size=size, vaccination_strategy=vaccination_strategy, graine=graine)
    #if lockdown:
    #    com.initiate_lockdown()
    
    press_add_person = st.button("Add person")
    if press_add_person:
        com.add_people(Person(id=len(com.population),status='infected', activity=.99, disease=disease, fragility=.5))
        st.write(f'{com.size} people in the community')
        
    step_size = st.number_input("Step size", 1, 60, 1,1)
    press_step = st.button("Step")
    if press_step:
        #for i in range(step_size):
        #    com.evolve()
        _ = [com.evolve() for _ in range(step_size)]
        st.write(f'R0: {round(com.R0(),2)}')
        st.write(com.plot_evolution())
        st.write(com.render_community_graph())
    #st.write(plt.plot(df[1:]))
    #st.write(com.render_community_graph())
    
if __name__ == '__main__':
        main()
