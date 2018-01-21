module MultivariateProblems

import Base.gradient

export UnconstrainedProblems

export OptimizationProblem, objective, gradient, objective_gradient, hessian

immutable OptimizationProblem{P, Tfg, Tf <: Real, TS <: AbstractString}
    name::TS
    f::Function
    g!::Function
    fg!::Tfg
    h!::Function
    constraints
    initial_x::Vector
    solutions::Vector
    minimum::Tf
    isdifferentiable::Bool
    istwicedifferentiable::Bool
    parameters::P
end

OptimizationProblem(name::AbstractString,
                    f::Function,
                    g!::Function,
                    fg!::Tfg,
                    h!::Function,
                    constraints,
                    initial_x::Vector,
                    solutions::Vector,
                    minimum::Tf,
                    isdifferentiable::Bool,
                    istwicedifferentiable::Bool) where Tf where Tfg =
                        OptimizationProblem(name, f, g!, fg!, h!, constraints,
                                            initial_x, solutions, minimum,
                                            isdifferentiable,
                                            istwicedifferentiable,
                                            nothing)

objective(p::OptimizationProblem{P}) where P<:Void = p.f
gradient(p::OptimizationProblem{P}) where P<:Void = p.g!
objective_gradient(p::OptimizationProblem{P}) where P<:Void = p.fg!
hessian(p::OptimizationProblem{P}) where P<:Void = p.h!

objective(p::OptimizationProblem{P}) where P = x-> p.f(x,p.parameters)
gradient(p::OptimizationProblem{P}) where P = (out,x)-> p.g!(out,x,p.parameters)
objective_gradient(p::OptimizationProblem{P}) where P = (out,x)-> p.fg!(out,x,p.parameters)
hessian(p::OptimizationProblem{P}) where P = (out,x)-> p.h!(out,x,p.parameters)

function objective_gradient(p::OptimizationProblem{P,Tfg}) where P where Tfg <: Void
    (out,x) -> begin
        gradient(p)(out,x)
        return objective(p)(x)
    end
end

function objective_gradient(p::OptimizationProblem{P,Tfg}) where P <: Void where Tfg <: Void
    (out,x) -> begin
        gradient(p)(out,x)
        return objective(p)(x)
    end
end


include("unconstrained.jl")
include("constrained.jl")

end
