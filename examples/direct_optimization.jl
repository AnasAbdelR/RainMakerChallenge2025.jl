using RainMakerChallenge2025
using OptimizationBBO, Optimization

function objective(params, p)
    return -max_precipitation(params)
end

lower_bounds = [0., -2000., 0., -180., -90., 270., 270., -5., -5., 5.]
upper_bounds = [2., 5000., 30., 180., 90., 300., 300., 5., 5., 50.]

optf = OptimizationFunction(objective)
prob = Optimization.OptimizationProblem(optf, [1, 0, 1, -80, 40.45, 300, 273, 0, 0, 35], 
                          lb=lower_bounds, ub=upper_bounds)

println("Starting optimization...")
sol = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(); maxiters=10_000)

println("Optimized parameters: ", sol.u)
println("Predicted max precipitation from surrogate: ", max_precipitation(sol.u))

#=
[1.9851412873870289, 2485.737125917869, 14.118878320714353, 73.66989910502846, -72.40588406991382, 280.9029383973223, 299.2432527466829, 4.773285514316327, 4.586811470190235, 33.398388938656204]
=#