
using CrystalNets
using JSON

CrystalNets.toggle_warning(false)
CrystalNets.toggle_export(false)

structure_type = "Auto"
if structure_type == "Zeolite"
    option = CrystalNets.Options(structure=CrystalNets.StructureType.Zeolite)
elseif structure_type == "MOF"
    option = CrystalNets.Options(structure=CrystalNets.StructureType.MOF)
else
    option = CrystalNets.Options(structure=CrystalNets.StructureType.Auto)
end

function process_one(cif_file)
    try
        result = CrystalNets.determine_topology(cif_file, option)
        output = []
        results_list = length(result) > 1 ? collect(result) : [result[1]]
        for res in results_list
            name = string(res[1])
            count = res[2]
            genome = res[1][CrystalNets.Clustering.Auto]
            dim = CrystalNets.ndims(CrystalNets.PeriodicGraph(genome))
            push!(output, Dict("dim" => dim, "name" => name, "count" => count))
        end
        return output
    catch e
        return Dict("error" => string(e))
    end
end

# Process all input files and return an array aligned to ARGS
function process_batch(files)
    results = Vector{Any}()
    for f in files
        push!(results, process_one(f))
    end
    return results
end

if length(ARGS) > 0
    println(JSON.json(process_batch(ARGS)))
end
