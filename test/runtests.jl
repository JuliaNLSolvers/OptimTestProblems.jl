using OptimTestProblems
using OptimTestProblems.MultivariateProblems

using Base.Test

verbose = false

@testset "Bounded univariate problems" begin
    uvp = OptimTestProblems.UnivariateProblems.examples
    for (name, p) in uvp
        verbose && print_with_color(:green, "Problem: $name \n")
        for (miz, mia) in zip(p.minimizers, p.minima)
            @test p.f(miz) ≈ mia
        end
    end

end


@testset "Unconstrained multivariate problems" begin
    muvp = MultivariateProblems.UnconstrainedProblems.examples
    for (name, p) in muvp
        verbose && print_with_color(:green, "Problem: $name \n")
        soltest = !any(isnan, p.solutions)

        if startswith(name, "Penalty Function I")
            # The provided solutions are not exact
            tol = 1e-16
        else
            tol = 1e-32
        end

        f = objective(p)
        soltest && @test f(p.solutions) ≈ p.minimum

        gs = similar(p.initial_x)
        g! = gradient(p)
        g!(gs, p.solutions)
        soltest && @test vecnorm(gs, Inf) < tol

        fg! = objective_gradient(p)
        fgs = similar(gs)
        g!(gs, p.initial_x)

        @test fg!(fgs, p.initial_x) ≈ f(p.initial_x)
        @test vecnorm(fgs.-gs, Inf)  < eps(eltype(gs))
    end
end
