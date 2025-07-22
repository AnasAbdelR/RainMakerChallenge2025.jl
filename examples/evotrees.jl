using JLSO
using EvoTrees
using OptimizationBBO, Optimization

path = joinpath(dirname(@__DIR__), "data", "100kdata.jlso")
data = JLSO.load(path)

x = data[:d].inputs
y = data[:d].outputs

X_train = x'
y_train = vec(y)

config = EvoTreeRegressor(
    nrounds=200,
    max_depth=4,
    eta=0.1,
    rng=123
)

println("Training EvoTrees model...")
model = EvoTrees.fit_evotree(config; x_train=X_train, y_train=y_train)
println("Model trained successfully")

function surrogate_model(params)
    pred = EvoTrees.predict(model, reshape(params, 1, :))
    return Float64(pred[1])
end

function objective(params, p)
    return -surrogate_model(params)
end

lower_bounds = [0., -2000., 0., -180., -90., 270., 270., -5., -5., 5.]
upper_bounds = [2., 5000., 30., 180., 90., 300., 300., 5., 5., 50.]

optf = OptimizationFunction(objective)
prob = Optimization.OptimizationProblem(optf, [1, 0, 1, -80, 40.45, 300, 273, 0, 0, 35], 
                          lb=lower_bounds, ub=upper_bounds)

println("Starting optimization...")
sol = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(); maxtime=180)

println("Optimized parameters: ", sol.u)
println("Predicted max precipitation from surrogate: ", surrogate_model(sol.u))

params = [1.9926539595097355, -1332.5917108103222, 0.12831000368229534, -179.9932141008324, -83.07773613043568, 274.57874370187, 274.71266497481173, 4.939867674866585, 4.833716894413193, 27.579215718327603]
precipitation = max_precipitation(params) # 142
