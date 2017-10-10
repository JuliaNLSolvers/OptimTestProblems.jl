using OptimTestProblems
using Base.Test

uvp = OptimTestProblems.UnivariateProblems.examples
for (name, p) in uvp
    for (miz, mia) in zip(p.minimizers, p.minima)
        @test p.f(miz) ≈ mia
    end
end
mvp = OptimTestProblems.UnconstrainedProblems.examples
for (name, p) in mvp
    gs = similar(p.initial_x)
    p.g!(gs, p.solutions)
    @test sum(gs) - zero(eltype(gs)) < 1e-32
end
