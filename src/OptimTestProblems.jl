module OptimTestProblems

export MultivariateProblems, UnivariateProblems


include("optim_tests/multivariate/multivariate.jl")
include("optim_tests/univariate/bounded.jl")


# Deprecation stuff
UnconstrainedProblems = begin
    Base.warn_once("UnconstrainedProblems is deprecated, use MultivariateProblems.UnconstrainedProblems instead.")
    OptimTestProblems.MultivariateProblems.UnconstrainedProblems
end

export UnconstrainedProblems

end # module
