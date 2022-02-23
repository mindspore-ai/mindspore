# multi_output.jl
module MultiOutput
export foo!

# inputs: a, b; outputs: c, d
function foo!(a, b, c, d)
    c .= a + b
    d .= a - b
end

end
