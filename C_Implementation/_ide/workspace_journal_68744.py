# 2025-09-12T17:10:24.606459
import vitis

client = vitis.create_client()
client.set_workspace(path="C_Implementation")

comp = client.create_hls_component(name = "BranchPredictionAI",cfg_file = ["hls_config.cfg"],template = "empty_hls_component")

comp = client.get_component(name="BranchPredictionAI")
comp.run(operation="SYNTHESIS")

comp.run(operation="SYNTHESIS")

vitis.dispose()

