### Performance tuning

x = 2

function f(y, n)
    for i in 1:n
        y = 0.4 * (x + y)
    end
    return y
end

@time f(3.0, 50_000_000)

@code_warntype f(3.0, 50_000_000)


function g(x, y, n)
    for i in 1:n
        y = 0.4 * (x + y)
    end
    return y
end

@time g(2, 3.0, 20_000_000)

@code_warntype g(2, 3.0, 100)
@code_native g(2, 3.0, 100)

# Lots more at https://docs.julialang.org/en/v1/manual/performance-tips/


### Visual profiler

@profview f(3.0, 10_000_000)
@profview g(2, 3.0, 10_000_000)


### Inspecting code

using SpecialFunctions

@which erfinv(0.2)

@edit erfinv(0.2)

@enter erfinv(0.2)


### Writing your own scripts, hot code reloading

include("auxiliary.jl")

A = randn(3, 3)

norm(custom_exponential(A) - exp(A))

using Revise

includet("auxiliary.jl")

norm(custom_exponential(A) - exp(A))

