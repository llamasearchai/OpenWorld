import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import logging

# Import the consolidated PhysicsConfig
# This path assumes that src/ is in PYTHONPATH or the package is installed.
from ...config import PhysicsConfig # Adjusted relative import path

logger = logging.getLogger(__name__)

class PhysicsEncoder(nn.Module):
    """Encodes input (e.g., frames or state) into a physics-relevant latent space."""
    def __init__(self, config: PhysicsConfig):
        super().__init__()
        self.config = config
        # Example: A simple MLP encoder. Replace with something more sophisticated.
        # Input dimension would depend on the nature of 'frames' or input state.
        # For now, assuming a flattened vector of some size (e.g., 1024 for placeholder)
        input_dim = 1024 # Placeholder, should be derived from actual input data spec
        self.fc = nn.Linear(input_dim, config.d_model)
        self.activation = getattr(F, config.activation, F.silu) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x might be raw frames (B, C, H, W) or preprocessed states (B, Seq, Dim)
        # Placeholder: assuming x is (Batch, InputDim)
        # Real implementation would need reshaping/conv layers if x is image data.
        if x.dim() > 2:
            x = x.view(x.size(0), -1) # Flatten
        
        # Simple check if input_dim matches fc layer, adjust if necessary (very basic)
        if x.shape[1] != self.fc.in_features:
            logger.warning(f"PhysicsEncoder input dim mismatch ({x.shape[1]} vs {self.fc.in_features}). This might lead to errors.")
            # Attempt to adapt with a new linear layer if drastically different (not ideal for production)
            # Or, more simply, ensure input is projected to the right size beforehand.

        encoded_state = self.activation(self.fc(x))
        return encoded_state

class PhysicsPredictor(nn.Module):
    """Predicts future physics states from current latent state."""
    def __init__(self, config: PhysicsConfig):
        super().__init__()
        self.config = config
        # Example: An MLP to predict next state or change in state.
        # This could be an RNN/Transformer if sequence prediction is needed.
        self.fc = nn.Linear(config.d_model, config.d_model) # Predicts next latent state
        self.activation = getattr(F, config.activation, F.silu)

    def forward(self, latent_state: torch.Tensor) -> torch.Tensor:
        # Predicts delta in latent_state or the next latent_state directly
        predicted_latent_delta = self.activation(self.fc(latent_state))
        # Example: return latent_state + predicted_latent_delta # If predicting delta
        return predicted_latent_delta # If predicting next state directly

class PhysicsDecoder(nn.Module): # Renamed from PhysicsRenderer for generality
    """Decodes latent physics state back to an observable state (e.g., frames, properties)."""
    def __init__(self, config: PhysicsConfig):
        super().__init__()
        self.config = config
        # Example: MLP to decode to some output (e.g., particle positions, image)
        # Output dimension depends on what needs to be rendered/decoded.
        output_dim = 3 # Placeholder for RGB or (x,y,z) positions
        self.fc = nn.Linear(config.d_model, output_dim) 
        # Activation might be tanh or sigmoid if output is bounded (e.g., image pixels)

    def forward(self, latent_state: torch.Tensor) -> torch.Tensor:
        decoded_output = self.fc(latent_state)
        # Example for image: decoded_output = torch.sigmoid(self.fc(latent_state))
        return decoded_output

class PhysicsEngine(nn.Module): # Renamed from PhysicsModule to PhysicsEngine
    """Neural physics engine for world modeling. Manages encoding, prediction, decoding, and loss calculation."""
    
    def __init__(self, config: PhysicsConfig):
        super().__init__()
        self.config = config
        
        self.encoder = PhysicsEncoder(config)
        self.predictor = PhysicsPredictor(config)
        self.decoder = PhysicsDecoder(config) # Renamed from renderer
        
        logger.info(f"PhysicsEngine initialized with d_model={config.d_model}, gravity={config.gravity_enabled}")

    def forward(self, current_obs: torch.Tensor, predict_steps: int = 1) -> Dict[str, torch.Tensor]:
        """
        Process current observation (e.g., frames or state vector) and predict future states/observations.
        Args:
            current_obs: Tensor representing the current observation.
                         Shape depends on input type (e.g., [B, C, H, W] for frames, [B, N, D] for particles).
            predict_steps: Number of future steps to predict.
        Returns:
            Dictionary containing input, predicted latent states, and decoded predictions.
        """
        if not isinstance(current_obs, torch.Tensor):
            raise TypeError(f"Input current_obs must be a torch.Tensor, got {type(current_obs)}")

        latent_state = self.encoder(current_obs)
        
        predicted_latent_states = []
        current_latent_for_pred = latent_state
        for _ in range(predict_steps):
            current_latent_for_pred = self.predictor(current_latent_for_pred)
            predicted_latent_states.append(current_latent_for_pred)
        
        # Decode the final predicted latent state, or all of them if needed
        # For simplicity, decoding the last one primarily
        predicted_observation = self.decoder(predicted_latent_states[-1]) if predicted_latent_states else self.decoder(latent_state)
        
        return {
            "input_observation": current_obs,
            "initial_latent_state": latent_state,
            "predicted_latent_states": torch.stack(predicted_latent_states) if predicted_latent_states else torch.empty(0),
            "predicted_observation": predicted_observation
        }
    
    def compute_loss(
        self, 
        predicted_values: torch.Tensor, # E.g., predicted particle positions, predicted frames
        ground_truth_values: Optional[torch.Tensor] = None, # Corresponding ground truth
        epoch_num: int = 0 # For logging or scheduled loss weights
    ) -> Dict[str, torch.Tensor]:
        """Compute physics-based losses and reconstruction losses."""
        # logger.info(f"Epoch {epoch_num}: Computing physics loss...")
        losses = {}

        # 1. Reconstruction Loss (if ground truth is available)
        reconstruction_loss = torch.tensor(0.0, device=predicted_values.device)
        if ground_truth_values is not None:
            if predicted_values.shape == ground_truth_values.shape:
                reconstruction_loss = F.mse_loss(predicted_values, ground_truth_values)
                losses["reconstruction_loss"] = reconstruction_loss
            else:
                logger.warning(
                    f"Shape mismatch for reconstruction loss. Predicted: {predicted_values.shape}, GT: {ground_truth_values.shape}. Skipping loss."
                )
                losses["reconstruction_loss"] = torch.tensor(0.0, device=predicted_values.device)
        else:
            losses["reconstruction_loss"] = torch.tensor(0.0, device=predicted_values.device)

        # 2. Physics Consistency Losses (placeholders, require actual physics logic)
        # These would operate on the predicted states (e.g., particle positions, velocities over time)
        # not just the final `predicted_values` which might be rendered frames.
        # This part is highly dependent on what `predicted_values` represents.
        
        # Placeholder for momentum conservation loss (if applicable)
        losses["momentum_loss"] = torch.tensor(0.0, device=predicted_values.device)
        # Placeholder for energy conservation loss (if applicable)
        losses["energy_loss"] = torch.tensor(0.0, device=predicted_values.device)
        # Placeholder for collision handling loss (if applicable)
        losses["collision_loss"] = torch.tensor(0.0, device=predicted_values.device)
        
        # Total physics-related loss (excluding reconstruction for now, or weighted sum)
        # This should be more than just the sum of placeholders if they are not implemented.
        physics_consistency_loss = losses["momentum_loss"] + losses["energy_loss"] + losses["collision_loss"]
        losses["physics_consistency_loss"] = physics_consistency_loss

        # Overall weighted loss
        total_loss = reconstruction_loss + self.config.conservation_loss_weight * physics_consistency_loss
        losses["total_loss"] = total_loss
        
        if epoch_num % 10 == 0: # Log less frequently
            log_msg = f"Epoch {epoch_num} Losses: Total={total_loss.item():.4f}, Recon={reconstruction_loss.item():.4f}, PhysicsCons={physics_consistency_loss.item():.4f}"
            logger.debug(log_msg)
            
        return losses 