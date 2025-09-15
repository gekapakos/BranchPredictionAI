# 2025-09-12T19:40:13.171500
import vitis

client = vitis.create_client()
client.set_workspace(path="C_Implementation")

comp = client.get_component(name="BranchPredictionAI")
comp.run(operation="C_SIMULATION")

comp.run(operation="SYNTHESIS")

vitis.dispose()

