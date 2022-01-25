# add.jl
module Add
export foo!

function my_func(x, y)
    return x + y
end

# z is output, should use . to inplace
function foo!(x, y, z)
    z .= my_func(x, y)
end

end
