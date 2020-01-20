# Get this figure: fig = py.get_figure("https://plot.ly/~terryollila/0/")
# Get this figure's data: data = py.get_figure("https://plot.ly/~terryollila/0/").get_data()
# Add data to this figure: py.plot(Data([Scatter(x=[1, 2], y=[2, 3])]), filename ="d3-cloropleth-map", fileopt="extend")
# Get z data of first trace: z1 = py.get_figure("https://plot.ly/~terryollila/0/").get_data()[0]["z"]

# Get figure documentation: https://plot.ly/python/get-requests/
# Add data documentation: https://plot.ly/python/file-options/

# If you're using unicode in your file, you may need to specify the encoding.
# You can reproduce this figure in Python with the following code!

# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

import plotly.plotly as py
from plotly.graph_objs import *
py.sign_in('username', 'api_key')
trace1 = {
  "uid": "90fb2c13-dc70-4d1b-a231-0cd7fe53e833", 
  "type": "choropleth", 
  "zsrc": "terryollila:1:292023", 
  "z": [37, 37, 36, 36, 34, 34, 33, 32, 31, 31, 30, 30, 30, 30, 29, 29, 29, 28, 28, 28, 27, 27, 27, 27, 27, 26, 26, 25, 25, 25, 25, 24, 24, 24, 24, 23, 23, 23, 22, 22, 21, 21, 20, 20, 19, 18, 17, 17, 16, 14], 
  "marker": {"line": {
      "color": "rgb(255,255,255)", 
      "width": 2
    }}, 
  "textsrc": "terryollila:1:1d686b", 
  "text": ["CO", "WA", "FL", "NV", "SD", "OR", "TN", "HI", "MN", "CA", "RI", "MA", "AZ", "ID", "DC", "MI", "OK", "NH", "SC", "ME", "MD", "GA", "UT", "NJ", "NE", "NY", "NC", "AR", "AK", "PA", "KY", "VA", "IA", "VT", "TX", "LA", "AL", "IN", "MO", "MT", "KS", "WV", "OH", "MS", "WI", "NM", "DE", "CT", "IL", "WY"], 
  "colorbar": {"title": {"text": "% Gain"}}, 
  "colorscale": [
    [0.0, "rgb(242,240,247)"], [0.2, "rgb(218,218,235)"], [0.4, "rgb(188,189,220)"], [0.6, "rgb(158,154,200)"], [0.8, "rgb(117,107,177)"], [1.0, "rgb(84,39,143)], 
  "locationmode": "USA-states", 
  "locationssrc": "terryollila:1:978052", 
  "locations": ["CO", "WA", "FL", "NV", "SD", "OR", "TN", "HI", "MN", "CA", "RI", "MA", "AZ", "ID", "DC", "MI", "OK", "NH", "SC", "ME", "MD", "GA", "UT", "NJ", "NE", "NY", "NC", "AR", "AK", "PA", "KY", "VA", "IA", "VT", "TX", "LA", "AL", "IN", "MO", "MT", "KS", "WV", "OH", "MS", "WI", "NM", "DE", "CT", "IL", "WY"], 
  "autocolorscale": False
}
data = Data([trace1])
layout = {
  "geo": {
    "scope": "usa", 
    "lakecolor": "rgb(255, 255, 255)", 
    "showlakes": True, 
    "projection": {"type": "albers usa"}
  }, 
  "title": {"text": "Forecasted Real Estate Growth by State<br>2018 - 2028"}
}
fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig)