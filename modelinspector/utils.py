import pandas
import plotly.express
import plotly.graph_objects
import torch.nn
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from plotly import graph_objects as go
from functools import reduce


def extract_hidden(model: torch.nn.Module, data: torch.Tensor):
    names = get_graph_node_names(model)[0]
    fe = create_feature_extractor(model, return_nodes=names)

    return fe(data)


def layer_to_str(layer: torch.nn.Module):
    return str(layer)


def model_to_chart(model: torch.nn.Module):
    names = get_graph_node_names(model)[0]
    data = list()

    for name in names:
        layer = None
        if "." in name:
            layer = reduce(getattr, name.split("."), model)
        else:
            layer = name

        data.append(layer_to_str(layer))

    fig = go.Figure(
        go.Funnel(
            x=data,
            y=names
        )
    )

    return fig


def model_to_sandkey(model: torch.nn.Module):
    names = get_graph_node_names(model)[0]  # use training names as some could be disabled during eval
    links = [{"source": i, "target": i + 1, "value": 1, "customdata": names[i]} for i in range(len(names) - 1)]
    data = {
        "nodes": names,
        "links": links
    }

    fig1 = go.Sankey(orientation="h",
                node=dict(
                    pad=10,
                    thickness=25,
                    line=dict(color="black", width=0.5),
                    label=data['nodes'],
                ),
                link=dict(
                    source=[link['source'] for link in data['links']],
                    target=[link['target'] for link in data['links']],
                    value=[link['value'] for link in data['links']],
                    customdata=[link['customdata'] for link in data['links']],
                    hovertemplate='%{source.customdata} to %{target.customdata}'
                )
            )

    fig = go.Figure(data=[fig1])

    return fig


def hidden_to_plotly(hidden: torch.Tensor, name: str = None) -> plotly.graph_objects.Figure:
    data = None
    if hidden.ndim == 4:  # conv2d
        data = plotly.graph_objects.Figure(
            plotly.express.imshow(hidden[0],
                                  animation_frame=0,
                                  labels=dict(animation_frame="channel"),
                                  title="<b>" + name + "</b>" if name is not None else "",
                                  color_continuous_scale="Greys_r",
                                  aspect="equal"
                                  ),
            layout={"autosize": True, "margin": {"b": 0, "t": 0, "r": 0, "l": 0}}
        )
        data.update_layout(title={'font': {"size": 20, 'family': "Arial Black"}})

    else:
        data = {
            "x": range(hidden.shape[1]),
            "y": hidden[0]
        }
        data = plotly.graph_objects.Figure(
            plotly.express.bar(data, x="x", y="y", title="<b>" + name + "</b>" if name is not None else "",
                               labels={"x": "neuron", "y": "Value"}),
            layout={"autosize": True, "margin": {"b": 0, "t": 0, "r": 0, "l": 0}}
        )
        data.update_layout(title={'font': {"size": 20, 'family': "Arial Black"}})

    return data
