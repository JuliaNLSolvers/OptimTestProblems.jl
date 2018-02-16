module OptimTestProblems

export MultivariateProblems, UnivariateProblems

include("optim_tests/multivariate/multivariate.jl")
include("optim_tests/univariate/bounded.jl")

# Deprecation stuff
UnconstrainedProblems = OptimTestProblems.MultivariateProblems.UnconstrainedProblems
export UnconstrainedProblems

end # module
