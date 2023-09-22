# можно делать таким подходом доски для отрисовки, но
# при этом прийдется подключаться к коспьютеру по сети, чтобы увидеть графики и остальное
# у себя в браузере...

from dash import Dash, dcc, html
import plotly.express as px
from base64 import b64encode
import io

app = Dash(__name__)

buffer = io.StringIO()

df = px.data.iris() # replace with your own data source
fig = px.scatter(
    df, x="sepal_width", y="sepal_length",
    color="species")
fig.write_html(buffer)

html_bytes = buffer.getvalue().encode()
encoded = b64encode(html_bytes).decode()

all_dash = html.Div([
    html.H4('Simple plot export options'),
    dcc.Graph(id="graph", figure=fig),
    html.A(
        html.Button("Download as HTML"),
        id="download",
        href="data:text/html;base64," + encoded,
        download="plotly_graph.html"
    ),
    dcc.Markdown('''

        *This text will be italic*

        _This will also be italic_


        **This text will be bold**

        __This will also be bold__

        _You **can** combine them_
    ''')
])
all_buffer = io.StringIO()

app.layout = all_dash

app.run_server(debug=True)
