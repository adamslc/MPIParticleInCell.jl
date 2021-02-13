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
    proc_dims::NTuple{N, IT}
end

function UniformGrid(global_lower_dimensions::NTuple{N, FT},
              global_upper_dimensions::NTuple{N, FT},
              global_num_cells::NTuple{N, IT},
              num_guard_cells::IT=zero(IT),
              periodic::NTuple{N, Bool}=ntuple(_ -> true, N),
             ) where {N, FT, IT}

    proc_dims = Tuple(compute_processor_dimensions(global_num_cells,
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
                           proc_dims
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
    compute_processor_dimensions(num_cells::NTuple{N}, num_procs)

Given the desired number of cells in each dimension, and a number of processors,
returns the number of subdivisions along each dimension that minimize the amount
of inter-rank communication.
"""
function compute_processor_dimensions(num_cells::NTuple{N}, num_procs) where N
    factors = factor(Vector, num_procs)

    proc_dims = fill(1, N)

    # For each prime factor of num_procs, we cut the simulation domain in the
    # dimension with the most cells remaining.
    for factor in Iterators.reverse(factors)
        # Find the current largest dimension
        largest_value = num_cells[1] / proc_dims[1]
        largest_index = 1
        for j in 2:N
            if num_cells[j] / proc_dims[j] > largest_value
                largest_value = num_cells[j] / proc_dims[j]
                largest_index = j
            end
        end

        # Cut along that dimension to make it smaller
        proc_dims[largest_index] *= factor
    end

    # We should have make exactly num_procs subdomains
    @assert reduce(*, proc_dims) == num_procs

    return proc_dims
end

"""
    compute_proc_indices(proc_dims, proc)

Computes the unique indices in the processor grid for `proc`.
"""
function compute_proc_indices(proc_dims, proc)
    indices = Vector{Int}(undef, length(proc_dims))
    factor = 1
    sum = 0
    for i in 1:length(proc_dims)
        indices[i] = div(mod(proc - sum, factor * proc_dims[i]), factor)
        factor *= proc_dims[i]
        sum += indices[i]
    end
    return Tuple(indices)
end

"""
    compute_proc_rank(proc_dims, indices)

Computes the rank associated with the processor at `indices`.
"""
function compute_proc_rank(proc_dims, indices)
    rank = 0
    prod = 1
    for i in 1:length(indices)
        rank += prod * mod(indices[i], proc_dims[i])
        prod *= proc_dims[i]
    end
    return rank
end

"""
    compute_proc_bounds(num_cells, proc_dims, proc)

Computes the range of cells that `proc` is responsible for.
"""
function compute_proc_bounds(num_cells, proc_dims, proc)
    indices = compute_proc_indices(proc_dims, proc)

    num_subcells = Tuple(div.(num_cells, proc_dims))
    extra_cells  = Tuple(mod.(num_cells, proc_dims))

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
    proc_dims = compute_processor_dimensions(num_cells, num_procs)

    lower_bounds = Vector{typeof(num_cells)}(undef, num_procs)
    upper_bounds = Vector{typeof(num_cells)}(undef, num_procs)

    for i in 1:num_procs
        lower_bounds[i], upper_bounds[i] = compute_proc_bounds(num_cells, proc_dims, i - 1)
    end

    return lower_bounds, upper_bounds
end

function decompose_domain(num_cells::NTuple{N}, num_procs, proc) where N
    proc_dims = compute_processor_dimensions(num_cells, num_procs)

    return compute_proc_bounds(num_cells, proc_dims, proc)
end
