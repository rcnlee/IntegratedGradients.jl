module IntegratedGradients

export gen_batch_data, 
    accuracy,
    predict,
    truth,
    seq_classifier_loss,
	integrate_gradient,
    check_condition,
    check_at_zero,
    check_at_one,
    check_at_x,
    OneHotVector_f,
    onehot_f,
    UniformHotVector_f,
    uniformhot_f

using MultivariateTimeSeries
using Flux
using Flux: onehot, batchseq, crossentropy, OneHotVector
using Base.Iterators

"""
symbols are discrete symbols
"""
function gen_batch_data{T}(mts::MTS, labels::Vector{T}, alphabet, batchsize::Int=50; 
                        b_normalize::Bool=true,
                        b_stop_token::Bool=true,
                        b_shuffle::Bool=true,
                        rng::AbstractRNG=Base.GLOBAL_RNG)

    mts = b_normalize ? normalize01(mts) : mts
    mts = b_stop_token ? append_stop_token(mts) : mts
    X = vec(mts)
    Y_ = [lab in alphabet ? onehot_f(lab, alphabet) : uniformhot_f(alphabet) for lab in labels]
    rp = collect(1:length(mts)) 
    if b_shuffle
        rp = randperm(rng, length(mts))
        X, Y_ = X[rp], Y_[rp]
    end
    Y = []
    for (x,y) in zip(X,Y_) 
        n = length(x)-1
        ymat = [zero(y) for i=1:n]
        push!(ymat, y)
        push!(Y, ymat)
    end
    Xs = collect(partition(X, batchsize))
    Ys = collect(partition(Y, batchsize))
    Xs = [batchseq(x,zero(x[1][1])) for x in Xs]
    Ys = [batchseq(y,zero(y[1][1])) for y in Ys] 

    Xs, Ys, rp
end

function accuracy(m, Xs, Ys)
    batchsize = size(Xs[1][1],2) 
    s = 0
    for (X,Y) in zip(Xs,Ys)
        Flux.reset!(m)
        Flux.truncate!(m)
        s += count(Flux.argmax(m.(X)[end]) .== Flux.argmax(Y[end]))
    end
    s / (batchsize*length(Xs))
end

function predict(m, Xs, ind) 
    Flux.reset!(m)
    Flux.truncate!(m)
	m.(Xs[ind])[end]
end
truth(Ys, ind) = Ys[ind][end]

function seq_classifier_loss(m)
    (X,Y) -> begin
        l = crossentropy(m.(X)[end],Y[end])
        Flux.truncate!(m)
        return l
    end
end

"""
See: Sundararajan, Taly, Yan, "Axiomatic attribution for deep networks"
Esp. equation 3.
Also see: github.com/ankurtaly/Attributions
"""
function integrate_gradient(m, X, target_index::Int, N::Int=100; skiplast::Bool=false)
    k = [i / N for i = 1:N] #assumes a zero baseline
    XP = [param(x * k') for x in X] 
    Flux.reset!(m)
    Flux.truncate!(m)
    y = m.(XP)[end][target_index, :]
    Flux.back!(y, 1)
    A = hcat([x .* mean(xp.grad, 2) for (x,xp) in zip(X,XP)]...) 
	return skiplast ? A[:, 1:(end-1)] : A
end

function check_condition(m, X, target_index::Int, N::Int=100; verbose=true)
    F0 = check_at_zero(m, X, target_index)
    F1 = check_at_x(m, X, target_index)

    A = integrate_gradient(m, X, target_index, N)
    err = abs(sum(A) - (F1-F0))
    verbose && println("Error is: $err")
    
    (err, sum(A), F1-F0)
end

function check_at_zero(m, X)
    XP0 = [zeros(x) for x in X] 
    Flux.reset!(m)
    Flux.truncate!(m)
    y0 = m.(XP0)[end]
    y0
end
function check_at_zero(m, X, target_index::Int)
    y0 = check_at_zero(m, X)[target_index,:]
    F0 = y0[1].tracker.data
    F0
end
function check_at_one(m, X)
    XP0 = [ones(x) for x in X] 
    Flux.reset!(m)
    Flux.truncate!(m)
    y0 = m.(XP0)[end]
    y0
end

function check_at_x(m, X)
    XP1 = [x for x in X] 
    Flux.reset!(m)
    Flux.truncate!(m)
    y1 = m.(XP1)[end]
    y1
end
function check_at_x(m, X, target_index::Int)
    y1 = check_at_x(m, X)[target_index,:]
    F1 = y1[1].tracker.data
    F1
end

struct OneHotVector_f <: AbstractVector{Float64}
    o::OneHotVector
end
Base.size(xs::OneHotVector_f) = size(xs.o)
Base.getindex(xs::OneHotVector_f, i) = Float64(getindex(xs.o, i))

onehot_f(l, labels) = OneHotVector_f(onehot(l,labels))
onehot_f(l, labels, unk) = OneHotVector_f(onehot(l,labels,unk))

struct UniformHotVector_f <: AbstractVector{Float64}
    sz::Int
end
Base.size(xs::UniformHotVector_f) = (xs.sz,)
Base.getindex(xs::UniformHotVector_f, i::Integer) = 1.0/xs.sz

uniformhot_f(labels) = UniformHotVector_f(length(labels))

end # module

