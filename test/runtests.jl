using MPIParticleInCell
using Test
using MPI

@testset "MPIParticleInCell.jl" begin
    MPI.Init()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    num_guard_cells = 1
    g = UniformGrid((0., 0.), (1., 1.), 2 .* (8, 5), num_guard_cells)
    rank == 0 && println(g)

    f = Field(g, "board", Int, true)
    rank == 0 && println(f)

    f.field_values .= rank
    gather_field_on_root!(f)
    rank == 0 && (display(f.field_values); println("\n"))

    f.field_values .= rank
    communicate_halo!(f, comm)
    rank == 0 && (display(f.field_values); println("\n"))

    MPI.Finalize()

    @test true
end
