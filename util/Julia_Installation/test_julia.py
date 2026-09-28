import subprocess
import json
from time import time

#julia_exe = os.environ["julia"]

# Create Julia script once
julia_script = """
using CrystalNets
using JSON

CrystalNets.toggle_warning(false)
CrystalNets.toggle_export(false)

function process_topology_to_json(cif_file)
    try
        option = CrystalNets.Options(structure=StructureType.Auto)
        result = CrystalNets.determine_topology(cif_file, option)
        
        # Convert to simple data structure
        output = []
        results_list = length(result) > 1 ? collect(result) : [result[1]]
        
        for res in results_list
            name = string(res[1])
            count = res[2]
            genome = res[1][CrystalNets.Clustering.Auto]
            dim = ndims(CrystalNets.PeriodicGraph(genome))
            push!(output, Dict("dim" => dim, "name" => name, "count" => count))
        end
        
        println(JSON.json(output))
    catch e
        println(JSON.json(Dict("error" => string(e))))
    end
end

if length(ARGS) > 0
    process_topology_to_json(ARGS[1])
end
"""

# Save the Julia script
with open("process_topology.jl", "w") as f:
    f.write(julia_script)

# Now use it
cif_file = "C.cif"

print("Calling Julia subprocess...")
t0 = time()

try:
    result = subprocess.run(
        ["julia", "process_topology.jl", cif_file],
        capture_output=True,
        text=True,
        timeout=600 # 30 second timeout
    )
    
    elapsed = time() - t0
    print(f"Call took: {elapsed:.2f}s")
    
    if result.returncode == 0:
        topology_data = json.loads(result.stdout.strip())
        print("Result:", topology_data)
        
        # Parse the result
        if "error" in topology_data:
            print("Error:", topology_data["error"])
        else:
            for topo in topology_data:
                print(f"  Dimension: {topo['dim']}, Name: {topo['name']}, Count: {topo['count']}")
    else:
        print("Julia error:", result.stderr)
        
except subprocess.TimeoutExpired:
    print(f"Timeout after 600 seconds!")
except Exception as e:
    print(f"Error: {e}")
