from typing import List
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class TextualDataMetrics:
    def __init__(self):
        pass 

    def plot_hist_tokens_pairs(self,src:List[List[str]],tgt:List[List[str]]):
        fig = make_subplots( 
            rows=1, cols=2, 
            subplot_titles=( 
                "Source tokens size Histogram", 
                "Target tokens size Histogram"
            )
        )

        fig.add_trace(
            go.Histogram(
                x=[len(tokens) for tokens in src]
            ),row=1,col=1
        )

        fig.add_trace(
            go.Histogram(
                x=[len(tokens) for tokens in tgt]               
            ), row=1,col=2
        )


        fig.show()