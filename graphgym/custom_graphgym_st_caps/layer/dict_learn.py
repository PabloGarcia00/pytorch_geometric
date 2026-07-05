import torch
import torch.nn as nn
from typing import Tuple

class DictionaryLearning(nn.Module):
    def __init__(self, feature_dim: int, n_atoms: int = 85,
                 lambda_sc: float = 0.1, max_iter: int = 10):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_atoms = n_atoms
        self.lambda_sc = lambda_sc
        self.max_iter = max_iter

        # Initialize dictionary atoms with unit norm
        self.dictionary = nn.Parameter(torch.randn(feature_dim, n_atoms))
        self.normalize_dictionary()

    def normalize_dictionary(self):
        with torch.no_grad():
            norms = torch.norm(self.dictionary, dim=0, keepdim=True)
            self.dictionary.data = self.dictionary.data / (norms + 1e-6)

    def update_dictionary(self, Z: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        N = Z.shape[0]
        device = Z.device
        Z_c = torch.matmul(Z.T, codes)
        c_c = torch.matmul(codes.T, codes)
        psi = torch.ones(self.n_atoms, device=device)
        Lambda = N * torch.diag(psi)

        for _ in range(5):
            D_new = torch.matmul(Z_c, torch.inverse(c_c + Lambda + 1e-6 * torch.eye(self.n_atoms, device=device)))
            atom_norms = torch.norm(D_new, dim=0)
            grad_psi = atom_norms ** 2 - 1
            psi = psi + 0.1 * grad_psi
            psi = torch.clamp(psi, min=0)
            Lambda = N * torch.diag(psi)

        D_new = torch.matmul(Z_c, torch.inverse(c_c + Lambda + 1e-6 * torch.eye(self.n_atoms, device=device)))
        norms = torch.norm(D_new, dim=0, keepdim=True)
        D_new = D_new / (norms + 1e-6)
        return D_new

    def update_codes(self, Z: torch.Tensor) -> torch.Tensor:
        N = Z.shape[0]
        device = Z.device
        D_T_D = torch.matmul(self.dictionary.T, self.dictionary)
        D_T_Z = torch.matmul(self.dictionary.T, Z.T)
        I = torch.eye(self.n_atoms, device=device)
        codes = torch.matmul(torch.inverse(D_T_D + self.lambda_sc * I + 1e-6 * I), D_T_Z).T

        for _ in range(self.max_iter):
            epsilon = 1e-6
            weights = 1.0 / (torch.abs(codes) + epsilon) ** 0.5
            W = torch.diag_embed(weights)
            codes_new = []
            for i in range(N):
                W_i = W[i]
                c_i = torch.matmul(torch.inverse(D_T_D + (self.lambda_sc / 2) * W_i + epsilon * I), self.dictionary.T @ Z[i])
                codes_new.append(c_i)
            codes = torch.stack(codes_new)
        return codes

    def forward(self, Z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        codes = self.update_codes(Z)
        reconstruction = torch.matmul(codes, self.dictionary.T)
        return codes, reconstruction


class SparseCodingModule(nn.Module):
    def __init__(self, feature_dim: int, n_atoms: int = 85, lambda_sc: float = 0.1):
        super().__init__()
        self.dict_learner = DictionaryLearning(feature_dim, n_atoms, lambda_sc)

    def forward(self, features: torch.Tensor, update_dict: bool = True) -> torch.Tensor:
        if features.dim() == 2:
            codes, _ = self.dict_learner(features)
            if update_dict and self.training:
                new_dict = self.dict_learner.update_dictionary(features, codes.detach())
                self.dict_learner.dictionary.data = new_dict
            return codes
        elif features.dim() == 3:
            T, N, M_prime = features.shape
            codes_list = []
            for t in range(T):
                codes_t, _ = self.dict_learner(features[t])
                codes_list.append(codes_t)
                if update_dict and self.training and t == T - 1:
                    new_dict = self.dict_learner.update_dictionary(features[t], codes_t.detach())
                    self.dict_learner.dictionary.data = new_dict
            return torch.stack(codes_list)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {features.shape}")
