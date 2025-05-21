#!/usr/bin/env python
"""
OpenWorld World Model Reasoning Example
====================================

This example demonstrates how to use the OpenWorld world model agent
for causal reasoning, counterfactual simulation, and pattern discovery.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Add parent directory to path if needed
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from openworld.agent.world_model_agent import WorldModelAgent, MetaWorldModelAgent, WorldModelStrategy, WorldState
from openworld.utils.logging import configure_logging, get_logger

# Configure logging
configure_logging(level="INFO")
logger = get_logger(__name__)

def run_example():
    """Run a world model reasoning example"""
    print("OpenWorld World Model Reasoning Example")
    print("====================================")
    
    # 1. Create a world model agent
    print("\nInitializing world model agent...")
    
    agent = MetaWorldModelAgent()
    
    # 2. Create a sample world state with entities and relationships
    print("\nCreating sample world state...")
    
    # Create a simple transportation scenario
    world_state = WorldState()
    
    # Add transportation entities
    world_state.add_entity("car1", {
        "type": "car",
        "color": "blue",
        "speed": 60,
        "position": [0, 0]
    })
    
    world_state.add_entity("bike1", {
        "type": "bike",
        "color": "red",
        "speed": 15,
        "position": [5, 2]
    })
    
    world_state.add_entity("road1", {
        "type": "road",
        "length": 10,
        "width": 2,
        "position": [0, 0]
    })
    
    world_state.add_entity("intersection1", {
        "type": "intersection",
        "traffic_light": "green",
        "position": [10, 0]
    })
    
    world_state.add_entity("pedestrian1", {
        "type": "pedestrian",
        "position": [9, 1]
    })
    
    # Add relationships
    world_state.add_relationship("car1", "road1", "on")
    world_state.add_relationship("bike1", "road1", "on")
    world_state.add_relationship("road1", "intersection1", "connects_to")
    world_state.add_relationship("pedestrian1", "intersection1", "near")
    
    # Add events with timestamps
    world_state.add_event("car_starts", {"entity": "car1"}, timestamp=0.0)
    world_state.add_event("bike_starts", {"entity": "bike1"}, timestamp=1.0)
    world_state.add_event("light_changes", {"entity": "intersection1", "from": "red", "to": "green"}, timestamp=5.0)
    world_state.add_event("car_stops", {"entity": "car1", "reason": "traffic"}, timestamp=8.0)
    world_state.add_event("pedestrian_crosses", {"entity": "pedestrian1"}, timestamp=10.0)
    
    # Update the agent with the world state
    agent.update_world_state(world_state)
    
    # 3. Visualize the world state as a graph
    print("\nVisualizing world state graph...")
    
    G = world_state.to_graph()
    pos = nx.spring_layout(G)
    
    plt.figure(figsize=(10, 8))
    
    # Draw nodes with different colors based on type
    node_colors = []
    for node in G.nodes():
        entity = world_state.entities.get(node, {})
        if entity.get("type") == "car":
            node_colors.append("skyblue")
        elif entity.get("type") == "bike":
            node_colors.append("salmon")
        elif entity.get("type") == "road":
            node_colors.append("grey")
        elif entity.get("type") == "intersection":
            node_colors.append("yellow")
        elif entity.get("type") == "pedestrian":
            node_colors.append("green")
        else:
            node_colors.append("lightgrey")
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=10)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrows=True)
    
    # Add edge labels
    edge_labels = {(u, v): d.get("type", "") for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("World Model - Transportation Scenario")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("world_model_graph.png", dpi=300)
    
    # 4. Perform causal reasoning
    print("\nPerforming causal reasoning...")
    
    causal_query = "Why did the car stop?"
    causal_result = agent.reason(causal_query, strategy=WorldModelStrategy.CAUSAL)
    
    print(f"\nQuery: {causal_query}")
    print("Inferences:")
    for i, inference in enumerate(causal_result["inferences"]):
        print(f"  {i+1}. {inference}")
    print(f"Confidence: {causal_result['confidence']:.2f}")
    
    # 5. Perform counterfactual reasoning
    print("\nPerforming counterfactual reasoning...")
    
    counterfactual_query = "What if the traffic light had remained red?"
    counterfactual_result = agent.reason(counterfactual_query, strategy=WorldModelStrategy.COUNTERFACTUAL)
    
    print(f"\nQuery: {counterfactual_query}")
    print("Inferences:")
    for i, inference in enumerate(counterfactual_result["inferences"]):
        print(f"  {i+1}. {inference}")
    print(f"Confidence: {counterfactual_result['confidence']:.2f}")
    
    # 6. Perform temporal reasoning
    print("\nPerforming temporal reasoning...")
    
    temporal_query = "What sequence of events occurred at the intersection?"
    temporal_result = agent.reason(temporal_query, strategy=WorldModelStrategy.TEMPORAL)
    
    print(f"\nQuery: {temporal_query}")
    print("Temporal Patterns:")
    for i, pattern in enumerate(temporal_result["patterns"]):
        print(f"  {i+1}. {pattern}")
    print(f"Confidence: {temporal_result['confidence']:.2f}")
    
    # 7. Try auto-selection of reasoning strategy
    print("\nDemonstrating automatic strategy selection...")
    
    queries = [
        "Why was the pedestrian at the intersection?",  # Should select causal
        "What if the bike was faster than the car?",    # Should select counterfactual
        "When did the light change to green?",          # Should select temporal
        "Where was the car positioned?",                # Should select spatial
        "How does the transportation system function?"  # Should select compositional
    ]
    
    print("\nAuto-selecting reasoning strategies:")
    for query in queries:
        strategy = agent.determine_best_strategy(query)
        print(f"  Query: {query}")
        print(f"  Selected strategy: {strategy}")
        print()
    
    # 8. Run a simple simulation
    print("\nRunning a simulation scenario...")
    
    simulation_scenario = {
        "name": "traffic_scenario",
        "entities": ["car1", "bike1", "pedestrian1"],
        "initial_conditions": {
            "car1": {"position": [0, 0], "speed": 60},
            "bike1": {"position": [5, 2], "speed": 15},
            "pedestrian1": {"position": [9, 1], "speed": 3}
        },
        "duration": 15,  # seconds
        "interventions": [
            {"time": 5, "entity": "intersection1", "action": "change_light", "value": "red"}
        ]
    }
    
    result = agent.simulate(simulation_scenario)
    print(f"\nSimulation result: {result['results']['analysis']}")
    
    # 9. Save results to a JSON file for later analysis
    print("\nSaving results to file...")
    
    all_results = {
        "causal": causal_result,
        "counterfactual": counterfactual_result,
        "temporal": temporal_result,
        "simulation": result
    }
    
    with open("world_model_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\nExample complete! Output files:")
    print("- world_model_graph.png")
    print("- world_model_results.json")

if __name__ == "__main__":
    run_example() 