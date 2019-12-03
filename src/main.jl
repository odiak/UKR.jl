using PyPlot, ForwardDiff, Random

function f(X, Y, X_in)
    D = dropdims(
        sum(
            (
                reshape(X_in, size(X_in, 1), 1, size(X_in, 2)) .-
                reshape(X, 1, size(X, 1), size(X, 2))
            ) .^ 2,
            dims = 3,
        ),
        dims = 3,
    )
    K = exp.(-0.5D)
    K_ = K ./ sum(K, dims = 2)
    K_ * Y
end

function estimate(Y, Q)
    N = size(Y, 1)
    X = randn(N, Q) * 0.05

    loss = X -> sum((Y - f(X, Y, X)) .^ 2) / N

    println(loss(X))
    for i in 1:300
        X .-= 60.0 * ForwardDiff.gradient(loss, X)
        println(loss(X))
    end

    X
end

function main()
    N = 100
    t = range(-1, 1, length = N)
    Y = zeros(N, 2)
    Y[:, 1] = t
    Y[:, 2] = sin.(π * t) * 0.5

    X = estimate(Y, 1)

    Y_estimated = f(X, Y, X)
    scatter(Y[:, 1], Y[:, 2])
    scatter(Y_estimated[:, 1], Y_estimated[:, 2])
    show()
end

main()
