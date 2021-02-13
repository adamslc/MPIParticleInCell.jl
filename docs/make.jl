using MPIParticleInCell
using Documenter

makedocs(;
    modules=[MPIParticleInCell],
    authors="Luke Adams <luke@lukeclydeadams.com> and contributors",
    repo="https://github.com/adamslc/MPIParticleInCell.jl/blob/{commit}{path}#L{line}",
    sitename="MPIParticleInCell.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://adamslc.github.io/MPIParticleInCell.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/adamslc/MPIParticleInCell.jl",
)
