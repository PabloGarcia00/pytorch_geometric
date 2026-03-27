import torch.optim as optim
from typing import Dict, List

def get_optimizer(model, config: Dict) -> List[optim.Optimizer]:
    optimizer_encoder_decoder = optim.Adam([
        {'params': model.encoder.parameters()},
        {'params': model.decoder.parameters()}
    ], lr=config['learning_rate'])

    optimizer_capsule = optim.Adam(
        model.capsule_regressor.parameters(),
        lr=config['learning_rate']
    )
    return [optimizer_encoder_decoder, optimizer_capsule]
