import plotly.graph_objects as go


def save_figure(fig, fn):
    fig.write_image(f"{fn}.png")
    fig.write_image(f"{fn}.pdf", format="pdf")


def add_layout(fig, x_label, y_label, title, font_size=25):
    fig.update_layout(
        template="none",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="center",
            x=0.5,
            itemsizing='constant'
        ),
        title=dict(
            text=title,
            font=dict(
                size=font_size
            )
        ),
        autosize=True,
        margin=go.layout.Margin(
            l=120,
            r=20,
            b=80,
            t=100,
            pad=0
        ),
        showlegend=True,
        xaxis=get_axis(x_label, font_size, font_size),
        yaxis=get_axis(y_label, font_size, font_size),
    )


def get_axis(title, title_size, tick_size):
    axis = dict(
        title=title,
        autorange=True,
        showgrid=True,
        zeroline=False,
        linecolor='black',
        showline=True,
        gridcolor='gainsboro',
        gridwidth=0.05,
        mirror="allticks",
        ticks='outside',
        titlefont=dict(
            color='black',
            size=title_size
        ),
        showticklabels=True,
        tickangle=0,
        tickfont=dict(
            color='black',
            size=tick_size
        ),
        exponentformat='e',
        showexponent='all'
    )
    return axis


def add_bar_trace(fig, x, y, text, name="", orientation='v'):
    showlegend = False if name == "" else True
    fig.add_trace(
        go.Bar(
            x=x,
            y=y,
            name=name,
            text=text,
            textposition='auto',
            showlegend=showlegend,
            orientation=orientation
        )
    )


def add_violin_trace(fig, y, name, showlegend=True):
    fig.add_trace(
        go.Violin(
            y=y,
            name=name,
            box_visible=True,
            meanline_visible=True,
            showlegend=showlegend,
        )
    )


def add_scatter_trace(fig, x, y, name, mode='markers', size=8):
    showlegend = False if name == "" else True
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            showlegend=showlegend,
            name=name,
            mode=mode,
            marker=dict(
                size=size,
                opacity=0.7,
                line=dict(
                    width=0.1
                )
            )
        )
    )
