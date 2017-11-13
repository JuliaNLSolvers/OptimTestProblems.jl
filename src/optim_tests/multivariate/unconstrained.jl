module UnconstrainedProblems

import Base.gradient

export OptimizationProblem, objective, gradient, hessian

#######################
# TODO: We could write fg! calls as well here
#######################
immutable OptimizationProblem{P,Tf <: Real}
    name::AbstractString
    f::Function
    g!::Function
    h!::Function
    initial_x::Vector{Tf}
    solutions::Vector
    minimum::Tf
    isdifferentiable::Bool
    istwicedifferentiable::Bool
    parameters::P
end

OptimizationProblem(name::AbstractString,
                    f::Function,
                    g!::Function,
                    h!::Function,
                    initial_x::Vector{Tf},
                    solutions::Vector,
                    minimum::Tf,
                    isdifferentiable::Bool,
                    istwicedifferentiable::Bool) where Tf =
                        OptimizationProblem(name, f, g!, h!, initial_x,
                                            solutions, minimum,
                                            isdifferentiable,
                                            istwicedifferentiable,
                                            nothing)

objective(p::OptimizationProblem{P}) where P<:Void = p.f
gradient(p::OptimizationProblem{P}) where P<:Void = p.g!
hessian(p::OptimizationProblem{P}) where P<:Void = p.h!

objective(p::OptimizationProblem{P}) where P = x-> p.f(x,p.parameters)
gradient(p::OptimizationProblem{P}) where P = (out,x)-> p.g!(out,x,p.parameters)
hessian(p::OptimizationProblem{P}) where P = (out,x)-> p.h!(out,x,p.parameters)

examples = Dict{AbstractString, OptimizationProblem}()

include("from_optim.jl")
include("quad_transforms.jl")
include("more_testing.jl")
end # module
