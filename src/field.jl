struct Field{ET, G, N}
    field_name::String
    field_values::Array{ET, N}
    grid::G
    full_field_on_root::Bool
end

function Field(grid::G, name::String, data_type::DataType=FT,
               full_field_on_root::Bool=false
              ) where {G <: UniformGrid{N, FT}} where {N, FT}
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    if full_field_on_root && rank == 0
        field_values = Array{data_type, N}(undef, grid.global_num_cells .+
                                           (2*grid.num_guard_cells + 1))
    else
        field_values = Array{data_type, N}(undef, grid.num_cells .+
                                           (2*grid.num_guard_cells + 1))
    end

    return Field{data_type, G, N}(name, field_values, grid, full_field_on_root)
end

Base.eltype(f::Field{ET, G}) where {ET, G} = ET

function Base.show(io::IO, f::Field{ET, G, N}) where {ET, G, N}
    print(io, "Field{$ET, $G, $N}($(f.grid), \"$(f.field_name)\")")
end

function Base.getindex(f::Field{ET, G, N}, indices...) where {ET, G, N}
    @assert length(indices) == N

    return getindex(f.field_values, (indices .+ f.grid.num_guard_cells)...)
end

function communicate_halo!(f::Field, comm)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    ngc = f.grid.num_guard_cells
    nc  = f.grid.num_cells

    indices = compute_proc_indices(f.grid.num_cuts, rank)
    onetos = map(Base.OneTo, nc .+ (2*ngc + 1))

    for (i, index) in enumerate(indices)
        lower_indices = tuple(indices[1:i-1]..., indices[i] - 1, indices[i+1:end]...)
        lower_proc_rank = compute_proc_rank(f.grid.num_cuts, lower_indices)
        upper_indices = tuple(indices[1:i-1]..., indices[i] + 1, indices[i+1:end]...)
        upper_proc_rank = compute_proc_rank(f.grid.num_cuts, upper_indices)

        sendrange1 = tuple(onetos[1:i-1]..., ngc + 1:2*ngc + 1,                 onetos[i+1:end]...)
        recvrange1 = tuple(onetos[1:i-1]..., ngc + nc[i] + 1:nc[i] + 2*ngc + 1, onetos[i+1:end]...)

        sview1 = view(f.field_values, sendrange1...)
        rview1 = view(f.field_values, recvrange1...)

        MPI.isend(sview1, lower_proc_rank, 10*i + 1, comm)
        data1, stat1 = MPI.recv(upper_proc_rank, 10*i + 1, comm)
        rview1 .= data1

        sendrange2 = tuple(onetos[1:i-1]..., nc[i] + 1:ngc + nc[i], onetos[i+1:end]...)
        recvrange2 = tuple(onetos[1:i-1]..., 1:ngc,             onetos[i+1:end]...)

        sview2 = view(f.field_values, sendrange2...)
        rview2 = view(f.field_values, recvrange2...)

        MPI.isend(sview2, upper_proc_rank, 10*i + 2, comm)
        data2, stat2 = MPI.recv(lower_proc_rank, 10*i + 2, comm)
        rview2 .= data2
    end
end

function gather_field_on_root!(f::Field)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    if rank == 0
        for proc in 1:size-1
            lbs, ubs = compute_proc_bounds(f.grid.global_num_cells, f.grid.num_cuts, proc)
            data, stat = MPI.recv(proc, 1, comm)
            indices = map((lb, ub) -> lb + f.grid.num_guard_cells:ub + f.grid.num_guard_cells, lbs, ubs)
            view(f.field_values, indices...) .= data
        end
    else
        ngc = f.grid.num_guard_cells
        indices = map(nc -> ngc + 1:ngc + nc, f.grid.num_cells)
        MPI.send(view(f.field_values, indices...), 0, 1, comm)
    end
end
