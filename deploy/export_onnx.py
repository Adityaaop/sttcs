import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import time

from core.models.gnn_encoder import GraphEncoder
from core.rl.maddpg_agent import Actor

class GraphEncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        
    def forward(self, x, edge_index):
        # Re-implementing the data unpack for ONNX tracing
        out = self.encoder.conv1(x, edge_index)
        out = self.encoder.activation(out)
        out = self.encoder.conv2(out, edge_index)
        out = self.encoder.activation(out)
        return torch.mean(out, dim=0, keepdim=True)

def main():
    print("Exporting Models to ONNX...")
    
    # 1. Initialize models
    encoder = GraphEncoder(node_in_dim=3, hidden_dim=64, embed_dim=128)
    encoder.eval()
    
    actor = Actor(obs_dim=5, embed_dim=128, action_dim=4)
    actor.eval()
    
    # 2. Wrapper for GNN to handle PyG Graph structure properly in ONNX
    gnn_wrapper = GraphEncoderWrapper(encoder)
    gnn_wrapper.eval()
    
    # Dummy inputs for GNN
    num_nodes = 10
    num_edges = 15
    dummy_x = torch.randn(num_nodes, 3)
    dummy_edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Export GNN
    torch.onnx.export(
        gnn_wrapper,
        (dummy_x, dummy_edge_index),
        "deploy/gnn_encoder.onnx",
        input_names=["x", "edge_index"],
        output_names=["graph_embed"],
        dynamic_axes={
            "x": {0: "num_nodes"},
            "edge_index": {1: "num_edges"}
        },
        opset_version=14
    )
    
    # Dummy inputs for Actor
    dummy_obs = torch.randn(1, 5)
    dummy_embed = torch.randn(1, 128)
    
    # Export Actor
    torch.onnx.export(
        actor,
        (dummy_obs, dummy_embed),
        "deploy/actor.onnx",
        input_names=["obs", "graph_embed"],
        output_names=["action"],
        dynamic_axes={
            "obs": {0: "batch_size"},
            "graph_embed": {0: "batch_size"},
            "action": {0: "batch_size"}
        },
        opset_version=14
    )
    print("Export Complete.")
    
    # 3. Validate Numerical Parity & Latency
    print("Validating ONNX Runtime Parity...")
    
    # PyTorch outputs
    with torch.no_grad():
        pt_embed = gnn_wrapper(dummy_x, dummy_edge_index)
        pt_action = actor(dummy_obs, dummy_embed)
        
    # ONNX Runtime execution
    ort_sess_gnn = ort.InferenceSession("deploy/gnn_encoder.onnx", providers=['CPUExecutionProvider'])
    ort_sess_actor = ort.InferenceSession("deploy/actor.onnx", providers=['CPUExecutionProvider'])
    
    # GNN Parity
    ort_inputs_gnn = {
        "x": dummy_x.numpy(),
        "edge_index": dummy_edge_index.numpy()
    }
    ort_embed = ort_sess_gnn.run(None, ort_inputs_gnn)[0]
    np.testing.assert_allclose(pt_embed.numpy(), ort_embed, rtol=1e-3, atol=1e-4)
    
    # Actor Parity
    ort_inputs_actor = {
        "obs": dummy_obs.numpy(),
        "graph_embed": dummy_embed.numpy()
    }
    ort_action = ort_sess_actor.run(None, ort_inputs_actor)[0]
    np.testing.assert_allclose(pt_action.numpy(), ort_action, rtol=1e-3, atol=1e-4)
    print("Numerical Parity Passed (tolerance < 1e-4).")
    
    # 4. Measure Inference Latency
    print("Measuring Latency (< 50ms Target)...")
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        embed = ort_sess_gnn.run(None, ort_inputs_gnn)[0]
        ort_sess_actor.run(None, {"obs": dummy_obs.numpy(), "graph_embed": embed})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
        
    avg_latency = sum(latencies) / len(latencies)
    print(f"Average Inference Latency (Batch=1): {avg_latency:.2f} ms")
    if avg_latency > 50.0:
        print("WARNING: Target latency of 50ms missed.")
    else:
        print("Latency target MET.")

if __name__ == "__main__":
    main()
