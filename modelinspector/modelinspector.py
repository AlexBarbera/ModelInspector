import torchvision.transforms as transforms
import dash
import plotly.express
import torch
import torchvision.models
from PIL import Image
from dash import Dash, html, Input, Output, dcc
import utils


def parse_args():
    import sys  # eww
    import argparse
    output = argparse.ArgumentParser()

    output.add_argument("model", type=str, help="Path to model.")
    output.add_argument("data", type=str, help="Path to tensor data.")

    return output.parse_args(sys.argv[1:])


def __init(args):
    model = torchvision.models.vgg16(torchvision.models.VGG16_Weights.DEFAULT)

    data = Image.open("assets/image.jpg")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    data = transform(data)
    data = data.reshape(1, *data.shape)

    mi = ModelInspector(model, data)
    mi.run()


class ModelInspector:
    def __init__(self, model: torch.nn.Module, data: torch.Tensor):
        if data.shape[0] > 1:
            raise Exception(
                "Found input with shape {}, expected single element batched tensor.".format(data.shape)
            )

        self.app = Dash("ModelInspector", external_stylesheets=["style.css"])
        self.model = model
        self.data = data
        self.host = None
        self.port = None
        self.debug = True
        self.meta = utils.extract_hidden(self.model, self.data)
        self.__init_app()

    def __init_app(self):
        self.app.layout \
            = html.Div(id="main",
                       children=[
                           html.H1("Model Inspector"),
                           html.Div(id="data",
                                    children=[
                                        dcc.Graph(id="model",
                                                  figure=utils.model_to_sandkey(self.model)
                                                  ),
                                        html.Div(id="images",
                                                 children=[
                                                     dcc.Graph(id="original",
                                                               figure=plotly.express.imshow(
                                                                   self.meta['x'].detach()[0].permute(1, 2, 0),
                                                                   title="Input",
                                                                   zmin=self.meta['x'].detach()[
                                                                       0].min().item(),
                                                                   zmax=self.meta['x'].detach()[
                                                                       0].max().item()
                                                               )
                                                               ),
                                                     dcc.Graph(id="layer",
                                                               figure=None
                                                               )
                                                 ])
                                    ])
                       ])

        @self.app.callback(Output(component_id="layer", component_property="figure"),
                           Input(component_id="model", component_property="hoverData"))
        def __render_layer(hover_info):
            #print("Callback with :", hover_info)
            if hover_info is None:
                return dash.no_update
            if hover_info["points"][0]["label"] == "":
                #print("Label is \'\'")
                return dash.no_update
            #print(self.meta[hover_info["points"][0]["label"]].detach().shape)

            fig = None

            if hover_info["points"][0]["label"] == 'x':
                #print(self.meta['x'].shape)
                x = self.meta['x'].detach()[0]
                fig = plotly.express.imshow(x.permute(1, 2, 0),
                                            title="Input", zmin=x.min().item(), zmax=x.max().item()
                                            )
            else:
                fig = utils.hidden_to_plotly(
                    self.meta[hover_info["points"][0]["label"]].detach(), name=hover_info["points"][0]["label"]
                )

            return fig

    def run(self):
        self.app.run(host=self.host, port=self.port, debug=self.debug)


if __name__ == "__main__":
    args = None  # parse_args()
    __init(args)
