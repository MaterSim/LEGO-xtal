
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
