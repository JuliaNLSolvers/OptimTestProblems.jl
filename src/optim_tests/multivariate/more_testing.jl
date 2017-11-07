### Source
###
### [3] Moré JJ, Garbow BS, Hillstrom KE: Testing unconstrained optimization software. ACM T Math Software. 1981
###

##########################
### Extended Rosenbrock
###
### Problem (21) from [3]
##########################

function extrosenbrock(x::AbstractArray, param::MatVecHolder)
    # TODO: we could do this without the xt storage holder
    n = length(x)
    jodd = 1:2:n-1
    jeven = 2:2:n

    xt = param.vec
    @. xt[jodd] = 10.0 * (x[jeven] - x[jodd]^2)
    @. xt[jeven] = 1.0 - x[jodd]

    return 0.5*sum(abs2, xt)
end

function extrosenbrock_gradient!(storage::AbstractArray,
                                 x::AbstractArray, param::MatVecHolder)
    n = length(x)
    jodd = 1:2:n-1
    jeven = 2:2:n
    xt = param.vec
    @. xt[jodd] = 10.0 * (x[jeven] - x[jodd]^2)
    @. xt[jeven] = 1.0 - x[jodd]

    @. storage[jodd] = -20.0 * x[jodd] * xt[jodd] - xt[jeven]
    @. storage[jeven] = 10.0 * xt[jodd]
end

function extrosenbrock_hessian!(storage,x,param)
    error("Hessian not implemented for Extended Hessian")
end

function _extrosenbrockproblem(N::Int;
                               initial_x::AbstractArray{T} = zeros(N),
                               name::AbstractString = "Extended Rosenbrock ($N)") where T
    OptimizationProblem(name,
                        extrosenbrock,
                        extrosenbrock_gradient!,
                        extrosenbrock_hessian!,
                        initial_x,
                        ones(initial_x),
                        true,
                        false,
                        MatVecHolder(Array{T}(0,0),similar(initial_x)))
end

examples["Extended Rosenbrock"] = _extrosenbrockproblem(100)
