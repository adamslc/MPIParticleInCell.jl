using Primes
using Printf

"""
Stores global and per-rank information about a grid imposed over a simulation
domain.
"""
abstract type AbstractGrid end

"""
    UniformGrid(lower_dims, upper_dims, num_cells, num_guard_cells, periodic)

Stores global and per-rank information about a uniform `N`-dimensional spatial
grid.
"""
struct UniformGrid{N, FT, IT} <: AbstractGrid
    global_num_cells::NTuple{N, IT}
    num_cells::NTuple{N, IT}
    lower_cell_bound::NTuple{N, IT}
    upper_cell_bound::NTuple{N, IT}
    num_guard_cells::IT

    global_lower_dimensions::NTuple{N, FT}
    lower_dimensions::NTuple{N, FT}
    global_upper_dimensions::NTuple{N, FT}
    upper_dimensions::NTuple{N, FT}
    global_simulation_size::NTuple{N, FT}
    simulation_size::NTuple{N, FT}
    cell_size::NTuple{N, FT}

    periodic::NTuple{N, Bool}
    num_cuts::NTuple{N, IT}
end

function UniformGrid(global_lower_dimensions::NTuple{N, FT},
              global_upper_dimensions::NTuple{N, FT},
              global_num_cells::NTuple{N, IT},
              num_guard_cells::IT=zero(IT),
              periodic::NTuple{N, Bool}=ntuple(_ -> true, N),
             ) where {N, FT, IT}

    num_cuts = Tuple(compute_decomposition_cuts(global_num_cells,
                                                MPI.Comm_size(MPI.COMM_WORLD)))

    lower_cell_bound, upper_cell_bound =
        decompose_domain(global_num_cells,
                         MPI.Comm_size(MPI.COMM_WORLD),
                         MPI.Comm_rank(MPI.COMM_WORLD))
    num_cells = upper_cell_bound .- lower_cell_bound .+ 1

    global_simulation_size  = global_upper_dimensions .- global_lower_dimensions
    cell_size = global_simulation_size ./ global_num_cells

    lower_dimensions = (lower_cell_bound .- 1) .* cell_size
    upper_dimensions = upper_cell_bound .* cell_size
    simulation_size  = upper_dimensions .- lower_dimensions

    return UniformGrid{N, FT, IT}(global_num_cells,
                           num_cells,
                           lower_cell_bound,
                           upper_cell_bound,
                           num_guard_cells,

                           global_lower_dimensions,
                           lower_dimensions,
                           global_upper_dimensions,
                           upper_dimensions,
                           global_simulation_size,
                           simulation_size,
                           cell_size,

                           periodic,
                           num_cuts
                          )
end

function Base.show(io::IO, g::UniformGrid{N, FT, IT}) where {N, FT, IT}
    print(io, "UniformGrid{$N, $FT, $IT}(global=(")
    for i in 1:N
        str = @sprintf("%.4g:%.4g:%.4g", g.global_lower_dimensions[i], g.cell_size[i],
                       g.global_upper_dimensions[i])
        print(io, str)
        i != N && print(io, ", ")
    end
    print(io, "), local=(")
    for i in 1:N
        str = @sprintf("%.4g:%.4g:%.4g", g.lower_dimensions[i], g.cell_size[i],
                       g.upper_dimensions[i])
        print(io, str)
        i != N && print(io, ", ")
    end
    print(io, ")), $(g.lower_cell_bound), $(g.upper_cell_bound)")
end

"""
    compute_decomposition_cuts(num_cells::NTuple{N}, num_procs)

Given the desired number of cells in each dimension, and a number of processors,
returns the number of subdivisions along each dimension that minimize the amount
of inter-rank communication.
"""
function compute_decomposition_cuts(num_cells::NTuple{N}, num_procs) where N
    factors = factor(Vector, num_procs)

    num_cuts = fill(1, N)

    # For each prime factor of num_procs, we cut the simulation domain in the
    # dimension with the most cells remaining.
    for factor in Iterators.reverse(factors)
        # Find the current largest dimension
        largest_value = num_cells[1] / num_cuts[1]
        largest_index = 1
        for j in 2:N
            if num_cells[j] / num_cuts[j] > largest_value
                largest_value = num_cells[j] / num_cuts[j]
                largest_index = j
            end
        end

        # Cut along that dimension to make it smaller
        num_cuts[largest_index] *= factor
    end

    # We should have make exactly num_procs subdomains
    @assert reduce(*, num_cuts) == num_procs

    return num_cuts
end

function compute_proc_indices(num_cuts, proc)
    indices = Vector{Int}(undef, length(num_cuts))
    factor = 1
    sum = 0
    for i in 1:length(num_cuts)
        indices[i] = div(mod(proc - sum, factor * num_cuts[i]), factor)
        factor *= num_cuts[i]
        sum += indices[i]
    end
    return Tuple(indices)
end

function compute_proc_rank(num_cuts, indices)
    rank = 0
    prod = 1
    for i in 1:length(indices)
        rank += prod * mod(indices[i], num_cuts[i])
        prod *= num_cuts[i]
    end
    return rank
end

function compute_proc_bounds(num_cells, num_cuts, proc)
    indices = compute_proc_indices(num_cuts, proc)

    num_subcells = Tuple(div.(num_cells, num_cuts))
    extra_cells  = Tuple(mod.(num_cells, num_cuts))

    lower_bound = num_subcells .* indices .+ 1 .+
        ifelse.(indices .< extra_cells, indices, extra_cells)
    upper_bound = lower_bound .+ num_subcells .- 1 .+
        ifelse.(indices .< extra_cells, 1, 0)

    return lower_bound, upper_bound
end

"""
    decompose_domain(num_cells::NTuple{N}, num_procs[, proc])

Given a target number of pieces `num_procs` and an `N`-cube simulation domain
with `num_cells` along each dimension, computes a decomposition of the domain.
If `proc` is specified, returns a pair of Tuples corresponding to the lower and
upper bounds of the cells assigned to `proc`. Otherwise, returns a pair of
vectors of the lower and upper bounds for each processor.

NOTE: `proc` is zero indexed.
"""
function decompose_domain(num_cells::NTuple{N}, num_procs) where N
    num_cuts = compute_decomposition_cuts(num_cells, num_procs)

    lower_bounds = Vector{typeof(num_cells)}(undef, num_procs)
    upper_bounds = Vector{typeof(num_cells)}(undef, num_procs)

    for i in 1:num_procs
        lower_bounds[i], upper_bounds[i] = compute_proc_bounds(num_cells, num_cuts, i - 1)
    end

    return lower_bounds, upper_bounds
end

function decompose_domain(num_cells::NTuple{N}, num_procs, proc) where N
    num_cuts = compute_decomposition_cuts(num_cells, num_procs)

    return compute_proc_bounds(num_cells, num_cuts, proc)
end
