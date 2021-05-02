#import csv
import pandas as pd
import plotly
import plotly.express as px
from plotly.offline import plot
import plotly.graph_objs as go
import numpy as np

class BotLog(object):
    def __init__(self):
      
        self.data=[{}]
        self.counter=0
    
    def logToCsv(self,csvName):
        self.csvName=csvName
    def tick(self):
        self.counter+=1
        self.data.append({})
    def log(self, message):
        # do plotting also here
        #if self.csvLogging:
            #self.csvLog(message)
        print(message)
        
    def collectData(self,snippet):
        
        for i in snippet.keys():
            
            self.data[self.counter][i]=snippet[i]
            
    def csvLog(self):
        self.df=pd.DataFrame(self.data)
        self.df.to_csv(self.csvName,header=True)
        self.df.replace(r'^\s*$', np.nan, regex=True,inplace=True)
        self.df.replace(0, np.nan, regex=False,inplace=True)
        #print(self.data)
        '''
        with open(self.csvName, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile,fieldnames=)
            #writer.writeheader()
            #writer.writerows(self.data)
            writer.writerows(self.data)
            self.data={}
        '''
    def histogram(self):
        if 'netYield' in self.df.columns:
            fig=px.histogram(data_frame=self.df,x="netYield")
            plot(fig)
    def scatter(self):
        '''
        def textcolor():
            #font_color = ['red' if v <= 0 else 'green' for v in self.df['netYield'] [pd.notna(self.df['netYield'])] ]
            font_color=[]
            for v in self.df['netYield']:
                if v<= 0:
                    font_color.append('red')
                elif v > 0:
                    font_color.append('green')
                else:
                    font_color.append('red')
            return font_color
            '''
        def annotation_list(string):
            annotations=[]
            for i in self.df[pd.notna(self.df[string])].index:
                annotations.append( 
                    dict(
                        x=self.df.loc[i,'date'],
                        y=self.df.loc[i,'close'],
                        xref="x",
                        yref="y",
                        text=self.df.loc[i,string].round(decimals=1),
                        showarrow=True,
                        arrowhead=7,
                        ax=0,
                        ay=-100,
                        font = dict(color= "red" if self.df.loc[i,string].round(decimals=1) <= 0 else  "green",
                                    family = 'sans serif',
                                    size = 14)
                        )
                )
            return annotations
        
                
        #font_color=textcolor()
        fig= plotly.subplots.make_subplots(rows=2, cols=1,shared_xaxes=True)
        dat_open=self.df[self.df['status']=='OPEN']
        dat_close=self.df[self.df['status']!='OPEN']
        #trace_all
        fig.add_trace(
            go.Scatter(
                x=self.df.date,
                y=self.df['close'],
                mode='lines+text',
                name='close',
            ),
            row=1, col=1
        )
        #trace_trade_open
        fig.add_trace(
            go.Scatter(
                x=dat_open.date,
                y=dat_open['close'],
                mode='markers',
                #text=dat_open.round({"netYield":2})['netYield'],
                name='Open',
                marker=dict(
                    size=5,
                    color='red'
                )
                
            ),
            row=1,col=1
        )
        #trace_trade_close
        fig.add_trace(
            go.Scatter(
                x=dat_close.date,
                y=dat_close['close'],
                mode='markers',
                #text=dat_close.round({"netYield":2})['netYield'],
                name='Closed',
                marker=dict(
                    size=5,
                    color='green' 
                )
                
            ),
            row=1,col=1
        )
        
        #trace_errors
        fig.add_trace(
            go.Scatter(
                x=self.df.date,
                y=self.df['close'],
                mode='markers',
                #text=dat_close.round({"netYield":2})['netYield'],
                name='Errorbar',
                opacity=1,
                marker=dict(
                    size=5,
                    #color=((self.df['status'] == 'CLOSED')).astype('int'),
                    #colorscale=[[0, 'purple'], [1, 'pink']]
                    color=['red' if self.df.loc[i,'status'] == 'OPEN' else 'green' for i in self.df.index]
                    #color= ["red" if self.df.loc[i,'netYield'].round(decimals=1) <= 0 else  "green" for i in self.df.index],
                            
                ),
                error_y=dict(
                type='data',
                    symmetric=False,
                    array=self.df['high'].to_numpy()-self.df['close'].to_numpy(),
                    arrayminus=-(self.df['low'].to_numpy()-self.df['close'].to_numpy()),
                    #color=['purple' if self.df.loc[i,'status'] == 'Closed' else 'pink' for i in self.df.index]
                    #color=((self.df['status'].to_numpy() == 'CLOSED')).astype('int'),
                    
                    #colorscale=[[0, 'red'], [1, 'green']]
                )
                
            ),
            row=1,col=1
        )   
        #trace_avg
        fig.add_trace(
            go.Scatter(
                x=self.df.date,
                y=self.df['movingAverage'],
                mode='lines',
                name='movingAverage',
                marker=dict(
                    size=2,
                    color='orange' 
                )
            ),
            row=1,col=1
        )
        #trace_mfi
        fig.add_trace(
            go.Scatter(
                x=self.df.date,
                y=self.df['MFI'],
                mode='lines',
                name='MFI',
                marker=dict(
                    size=2,
                    color='purple' 
                )
            ),
            row=2,col=1
        )
        '''
                color=(
                    (self.df.status == "OPEN") #dfp&(self.df.y < self.df.upper_limit)
                    ).astype('int'),
                colorscale=[[1, 'blue'], [0, 'green']],
        '''
        
        #data = [trace_all,trace_open, trace_close,trace_avg,trace_mfi]
        '''
        layout = go.Layout(
        barmode='group',
        legend=dict(orientation="h")
        )
        '''
        
        #fig = go.Figure(data=data, layout=layout)
        if 'netYield' in self.df.columns:
            fig.update_layout(
                annotations=annotation_list('netYield'),
                 xaxis=dict(
                     rangeslider=dict(
                         visible=True
                ),
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6m",
                             step="month",
                             stepmode="backward"),
                        dict(count=1,
                             label="YTD",
                             step="year",
                             stepmode="todate"),
                        dict(count=1,
                             label="1y",
                             step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                type="date"
                )
            )
        
        
        #fig.for_each_trace(textcolor)
        
        plot(fig)
        
        