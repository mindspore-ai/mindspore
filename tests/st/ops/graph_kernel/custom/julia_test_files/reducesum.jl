# reducesum.jl
module ReduceSum
export foo!
include("row_major.jl")

function foo!(x, y)
    x = change_input_to_row_major(x)
    # julia axis = 2 equals numpy axis = 1
    y .= sum(x, dims=2)
    y .= change_output_to_row_major(y)
end

end
