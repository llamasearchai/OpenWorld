from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
import logging
import time

logger = logging.getLogger(__name__)

# Assuming WorldState definition might come from a shared types module or agent.core later
# For now, keeping it simple or assuming it's implicitly defined by usage.
# from ..core import WorldState # If WorldState is defined in agent.core

@dataclass
class WorldStateSnapshot:
    timestamp: float
    description: str
    state_data: Dict[str, Any] # A copy of the world state at that time

class WorldStateMemory:
    """Manages the agent's understanding of the world state over time."""
    
    def __init__(self, max_history_snapshots: int = 100):
        self.current_state: Dict[str, Any] = {
            "entities": {},
            "relationships": [],
            "events": []
        } # This structure should align with WorldState/WorldKnowledgeState
        self.history: List[WorldStateSnapshot] = []
        self.max_history_snapshots = max_history_snapshots
        logger.info(f"WorldStateMemory initialized. Max history: {max_history_snapshots}")

    def update_state(self, update_data: Dict[str, Any]):
        """Updates the current world state based on new information."""
        if not isinstance(update_data, dict):
            logger.warning("update_state received non-dict data. Ignoring.")
            return

        if "entities" in update_data and isinstance(update_data["entities"], dict):
            # Deep merge for entities to update existing ones or add new ones
            for entity_id, entity_props in update_data["entities"].items():
                if entity_id in self.current_state["entities"] and isinstance(self.current_state["entities"][entity_id], dict):
                    self.current_state["entities"][entity_id].update(entity_props)
                else:
                    self.current_state["entities"][entity_id] = entity_props
        
        if "relationships" in update_data and isinstance(update_data["relationships"], list):
            # Could add checks for duplicates or update existing relationships if identifiable
            self.current_state["relationships"].extend(update_data["relationships"])
        
        if "events" in update_data and isinstance(update_data["events"], list):
            self.current_state["events"].extend(update_data["events"])
        
        logger.debug(f"World state updated. Entities: {len(self.current_state['entities'])}, Events: {len(self.current_state['events'])}")

    def get_current_state(self) -> Dict[str, Any]:
        """Returns a deep copy of the current world state."""
        # Return a copy to prevent external modification of the internal state directly
        return json.loads(json.dumps(self.current_state)) # Simple deep copy via JSON serialization

    def get_current_state_summary(self) -> Dict[str, Any]:
        """Returns a brief summary of the current state."""
        return {
            "num_entities": len(self.current_state.get("entities", {})),
            "num_relationships": len(self.current_state.get("relationships", [])),
            "num_events": len(self.current_state.get("events", [])),
            "last_event_timestamp": self.current_state.get("events", [])[-1].get("timestamp") if self.current_state.get("events") else None
        }

    def commit_state(self, description: str) -> WorldStateSnapshot:
        """Saves the current state as a snapshot in history."""
        if len(self.history) >= self.max_history_snapshots:
            self.history.pop(0) # Remove oldest snapshot if max capacity reached
        
        snapshot_data = self.get_current_state() # Get a deep copy
        snapshot = WorldStateSnapshot(
            timestamp=time.time(),
            description=description,
            state_data=snapshot_data
        )
        self.history.append(snapshot)
        logger.info(f"Committed world state snapshot: '{description}'. History size: {len(self.history)}")
        return snapshot

    def get_snapshot(self, timestamp: Optional[float] = None, description_match: Optional[str] = None) -> Optional[WorldStateSnapshot]:
        """Retrieves a specific snapshot. Prioritizes timestamp if both given."""
        if timestamp is not None:
            # Find closest snapshot by timestamp (could be more sophisticated)
            closest_snapshot = min(self.history, key=lambda s: abs(s.timestamp - timestamp), default=None)
            return closest_snapshot
        elif description_match is not None:
            for snapshot in reversed(self.history): # Search newest first
                if description_match.lower() in snapshot.description.lower():
                    return snapshot
        return None # Or raise error if not found

    def query_memory(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Queries the current state or historical snapshots (placeholder)."""
        # This is a placeholder for a more advanced query system.
        # For example, could query entities by property, events by type/time range, etc.
        logger.warning("Basic query_memory called. Advanced querying not yet implemented.")
        results = []
        if "entity_id" in query and query["entity_id"] in self.current_state["entities"]:
            results.append(self.current_state["entities"][query["entity_id"]])
        return results

    def clear_memory(self):
        """Resets the current state and history."""
        self.current_state = {"entities": {}, "relationships": [], "events": []}
        self.history = []
        logger.info("WorldStateMemory cleared.")

# Potential for QuantumStateMemory if it's a distinct concept for superposition of states, etc.
# class QuantumStateMemory(WorldStateMemory):
#     def __init__(self, max_history_snapshots: int = 100):
#         super().__init__(max_history_snapshots)
#         logger.info("QuantumStateMemory initialized.")
#     # Override or add methods specific to quantum-like state management
#     # e.g., handling superpositions, probabilities of states. 