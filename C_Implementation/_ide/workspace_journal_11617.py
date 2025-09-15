# 2025-09-13T18:11:42.746431
import vitis

client = vitis.create_client()
client.set_workspace(path="C_Implementation")

comp = client.get_component(name="BranchPredictionAI")
comp.run(operation="SYNTHESIS")

comp.run(operation="SYNTHESIS")

vitis.dispose()

