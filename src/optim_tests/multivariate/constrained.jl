module ConstrainedProblems

using ..OptimizationProblem,..ConstraintData

examples = Dict{AbstractString, OptimizationProblem}()

hs9_obj(x::AbstractVector) = sin(π*x[1]/12) * cos(π*x[2]/16)
hs9_c!(c::AbstractVector, x::AbstractVector) = (c[1] = 4*x[1]-3*x[2]; c)
hs9_h!(h, x, λ) = h

function hs9_obj_g!(g::AbstractVector, x::AbstractVector)
    g[1] = π/12 * cos(π*x[1]/12) * cos(π*x[2]/16)
    g[2] = -π/16 * sin(π*x[1]/12) * sin(π*x[2]/16)
    g
end
function hs9_obj_h!(h::AbstractMatrix, x::AbstractVector)
    v = hs9_obj(x)
    h[1,1] = -π^2*v/144
    h[2,2] = -π^2*v/256
    h[1,2] = h[2,1] = -π^2 * cos(π*x[1]/12) * sin(π*x[2]/16) / 192
    h
end

function hs9_jacobian!(J, x)
    J[1,1] = 4
    J[1,2] = -3
    J
end

examples["HS9"] = OptimizationProblem("HS9",
                                      hs9_obj,
                                      hs9_obj_g!,
                                      nothing,
                                      hs9_obj_h!,
                                      ConstraintData(hs9_c!, hs9_jacobian!, hs9_h!,
                                                     [], [], [0.0], [0.0]),
                                      [0.0, 0.0],
                                      [-3,-4],#[[12k-3, 16k-4] for k in (0, 1, -1)], # any integer k will do...
                                      hs9_obj([-3.0,-4.0]),
                                      true,
                                      true)


end  # module
