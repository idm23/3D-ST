"""File containing torch models
"""
# ======== standard imports ========
# ==================================

# ======= third party imports ======
import torch
import torch.nn as nn
import numpy as np
# ==================================

# ========= program imports ========
import st3d.consts as consts
import st3d.analytical as anly
# ==================================

class SharedMLP(nn.Module):
    def __init__(
        self,
        input_dim:int,
        output_dim:int,
    ) -> None:
        super(SharedMLP, self).__init__()

        self.dense = nn.Linear(input_dim, output_dim)
        self.activation_fn = nn.LeakyReLU(0.2)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.activation_fn(self.dense(x))
        return x

class LFA(nn.Module):
    def __init__(self, d_lfa:int) -> None:
        super(LFA, self).__init__()
        self.mlp = SharedMLP(4, d_lfa)

    def forward(
            self, 
            input_features: torch.Tensor, 
            geom_features: torch.Tensor, 
            closest_indices: torch.Tensor
        ) -> torch.Tensor:
        # Shape: geom_features (N, k, 4), input_features (N, d_lfa)
        geom_outputs = self.mlp(geom_features)  # Shape (N, k, d_lfa)
    

        # Gather the relevant input features using indices
        relevant_input_features = input_features[closest_indices]  # Shape  (N, k, d_lfa)

        # Concatenate along the last dimension
        combined_features = torch.cat((geom_outputs, relevant_input_features), dim=-1)  # Shape (N, k, 2*d_lfa)
        # Avg Pool the neighbors
        output_features = torch.mean(combined_features, dim = 1)

        return output_features
        
class Residual(nn.Module):
    def __init__(self, d:int) -> None:
        super(Residual, self).__init__()

        self.mlp_initial = SharedMLP(d, d//4)
        self.lfa1 = LFA(d//4)
        self.lfa2 = LFA(d//2)
        self.mlp_final = SharedMLP(d, d)
        self.mlp_resiudal = SharedMLP(d, d)

    def forward(
            self, 
            input_features:torch.Tensor, 
            geom_features:torch.Tensor, 
            closest_indices:torch.Tensor
        ) -> torch.Tensor:
        residual = self.mlp_resiudal(input_features) # Shape(N, d)
        activated = self.mlp_initial(input_features) # Shape(N, d//4)
        activated = self.lfa1(activated, geom_features, closest_indices) # Shape(N, d//2)
        activated = self.lfa2(activated, geom_features, closest_indices) # Shape(N, d)
        activated = self.mlp_final(activated) # Shape(N, d)
        return activated + residual
    
class StudentTeacher(nn.Module):
    def __init__(self, k:int, d:int, num_residuals:int) -> None:
        super(StudentTeacher, self).__init__()

        self.k = k
        self.d = d
        self.residuals = nn.ModuleList(
            [Residual(d) for i in range(num_residuals)]
        )

    def forward(
            self, 
            points:torch.Tensor, 
            geom_features:torch.Tensor, 
            closest_indices:torch.Tensor
        ) -> torch.Tensor:
        # Points: N, 3
        point_features = torch.zeros((points.shape[0], self.d)).to(consts.DEVICE) # Shape (N, d)
        for residual_layer in self.residuals:
            point_features = residual_layer(point_features, geom_features, closest_indices) # Shape (N, d)
        return point_features
    
    def point_based_forward(self, points:torch.Tensor) -> torch.Tensor:
        geom_features, closest_indices = anly.calculate_geom_features(points, self.k) # Shape (N, k, 4), (N, k)
        return self(points, geom_features, closest_indices)

    
class Decoder(nn.Module):
    def __init__(self, d:int) -> None:
        super(Decoder, self).__init__()

        self.input_layer = SharedMLP(d, consts.HIDDEN_LAYER_D)
        self.hidden_1 = nn.Sequential(*[
            nn.Linear(consts.HIDDEN_LAYER_D, consts.HIDDEN_LAYER_D), 
            nn.LeakyReLU(consts.HIDDEN_LAYER_LEAKY_SLOPE)
        ])
        self.hidden_2 = nn.Sequential(*[
            nn.Linear(consts.HIDDEN_LAYER_D, consts.HIDDEN_LAYER_D), 
            nn.LeakyReLU(consts.HIDDEN_LAYER_LEAKY_SLOPE)
        ])
        self.output_layer = nn.Linear(consts.HIDDEN_LAYER_D, consts.NUM_DECODED_POINTS*3)

    def forward(self, point_features:torch.Tensor) -> torch.Tensor:
        x = self.input_layer(point_features)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        return self.output_layer(x).reshape(-1, consts.NUM_DECODED_POINTS, 3)

if __name__ == "__main__":
    point_cloud_0 = np.load('point_clouds/point_cloud_0.npy')
    point_cloud_0 = torch.from_numpy(point_cloud_0).float().to(consts.DEVICE)
    print(point_cloud_0, point_cloud_0.shape)
    teacher = StudentTeacher(consts.K, consts.D, consts.NUM_RESIDUALS).to(consts.DEVICE)
    point_features = teacher.point_based_forward(point_cloud_0)
    print(point_features.shape)



        
