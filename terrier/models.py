from torch import nn

# class FinalLayer(nn.Module):
#     def __init__(
#         self,
#         layer_to_substitute:nn.Module,
#     ):
#         final_in_features = list(layer_to_substitute.modules())[1].in_features
#         self.penultimate = nn.Linear(in_features=final_in_features, out_features=final_in_features, bias=True),
#         nn.Linear(in_features=final_in_features, out_features=output_size, bias=final_bias)

#         nn.Sequential(
#             nn.Linear(in_features=final_in_features, out_features=final_in_features, bias=True),
#             nn.ReLU(),
#             ,
#         )    


class VectorOutput(nn.Module): 
    def __init__(self, final, **kwargs):
        super().__init__(**kwargs)
        self.final = final
        self.penultimate = list(final.modules())[1]

    def forward(self, x):
        result = self.final(x)
        return result, self.penultimate(x)
