module UnconstrainedProblems

using ..OptimizationProblem

export OptimizationProblem

import ..objective, ..gradient, ..hessian

examples = Dict{AbstractString, OptimizationProblem}()

include("from_optim.jl")
include("quad_transforms.jl")
include("more_testing.jl")
end # module
