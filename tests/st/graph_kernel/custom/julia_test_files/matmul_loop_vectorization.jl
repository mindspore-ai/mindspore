# matmul_loop_vectorization.jl
module Matmul
export foo!
include("row_major.jl")

# if dont have LoopVectorization pkg, install it as below
# import Pkg
# Pkg.add("LoopVectorization")
using LoopVectorization

function gemmavx(x, y, z)
    @turbo for m ∈ axes(x, 1), n ∈ axes(y, 2)
        zmn = zero(eltype(z))
        for k ∈ axes(x, 2)
            zmn += x[m, k] * y[k, n]
        end
        z[m, n] = zmn
    end
    return z
end

# z is output, should use . to inplace
# julia array is column-major, numpy aray is row-major
# user should transpose julia or numpy's array to keep same behavior
function foo!(x, y, z)
    x = change_input_to_row_major(x)
    y = change_input_to_row_major(y)
    z .= gemmavx(x, y, z)
    z .= change_output_to_row_major(z)
end

end
