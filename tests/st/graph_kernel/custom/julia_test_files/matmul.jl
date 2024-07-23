# matmul.jl
module Matmul
export foo!
include("row_major.jl")

function gemm(x, y, z)
    @inbounds @fastmath for m ∈ axes(x, 1), n ∈ axes(y, 2)
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
# user should change julia or numpy's layout to keep same behavior
#= EXAMPLE
A[2,3]               B[3,4]               C[2,4]
NUMPY:
[[1, 2, 3]       [[1, 2, 3, 4]         [[38, 44, 50,  56]
 [4, 5, 6]]       [5, 6, 7, 8]          [83, 98, 113,128]]
                  [9,10,11,12]]
JULIA:
inputs read numpy data from memory:
[[1, 3, 5]       [[1, 4, 7,10]
 [2, 4, 6]]       [2, 5, 8,11]
                  [3, 6, 9,12]]
inputs after reshape(reverse(shape)):
[[1, 4]          [[1, 5, 9]
 [2, 5]           [2, 6,10]
 [3, 6]]          [3, 7,11]
                  [4, 8,12]]
inputs after transpose/permutedims:
[[1, 2, 3]       [[1, 2, 3, 4]         [[38, 44, 50,  56]
 [4, 5, 6]]       [5, 6, 7, 8]          [83, 98, 113,128]]
                  [9,10,11,12]]
output after transpose/permutedims:
                                       [[38, 83]
                                        [44, 98]
                                        [50,113]
                                        [56,128]
output after reshape:
                                       [[38, 50, 83, 113]
                                        [44, 56, 98, 128]]
output read numpy data from memory:
                                       [[38, 44, 50,  56]
                                        [83, 98,113, 128]]
=#
function foo!(x, y, z)
    x = change_input_to_row_major(x)
    y = change_input_to_row_major(y)
    z .= gemm(x, y, z)
    z .= change_output_to_row_major(z)
end

end
