module MPIParticleInCell

using MPI

export UniformGrid, Field, gather_field_on_root!, communicate_halo!

include("grid.jl")
include("field.jl")

end
