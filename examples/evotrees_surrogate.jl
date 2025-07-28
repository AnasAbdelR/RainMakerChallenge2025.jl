using JLSO
using EvoTrees
using OptimizationBBO, Optimization

path = joinpath(dirname(@__DIR__), "data", "10kdata.jlso")
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
sol = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(); maxtime=1000)

println("Optimized parameters: ", sol.u)
println("Predicted max precipitation from surrogate: ", surrogate_model(sol.u))

params = [1.9838670323926768, -829.5647051140496, 0.37369094997301183, -179.95542817166884, -85.76397590364414, 274.4111266309889, 299.97239711713456, 4.99088309704113, 4.929389013671536, 27.617798405027]
precipitation = max_precipitation(params)

params = [1.9926539595097355, -1332.5917108103222, 0.12831000368229534, -179.9932141008324, -83.07773613043568, 274.57874370187, 274.71266497481173, 4.939867674866585, 4.833716894413193, 27.579215718327603]
precipitation = max_precipitation(params) # 142

params = [1.974439945381504, -886.5327716537533, 0.20726023142291083, -177.45775355613495, -88.18073023018059, 274.26019700021243, 299.9999139429991, 4.622875177595963, 4.962707029468203, 27.75239493195939]
precipitation = max_precipitation(params)