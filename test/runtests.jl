using OptimTestProblems
using Base.Test

uvp = OptimTestProblems.UnivariateProblems.examples
for (name, p) in uvp
    for (miz, mia) in zip(p.minimizers, p.minima)
        @test p.f(miz) ≈ mia
    end
end

using OptimTestProblems.UnconstrainedProblems
mvp = OptimTestProblems.UnconstrainedProblems.examples
for (name, p) in mvp
    if any(isnan, p.solutions)
        # TODO: how to check these problems?
        continue
    end
    f = objective(p)
    gs = similar(p.initial_x)
    pg! = gradient(p)
    pg!(gs, p.solutions)
    @show name, p.minimum
    if startswith(name, "Penalty Function I")
        tol = 1e-16
    else
        tol = 1e-32
    end

    @test norm(gs, Inf) < tol
    @test f(p.solutions) ≈ p.minimum
end
