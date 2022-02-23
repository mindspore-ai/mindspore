export change_input_to_row_major
export change_output_to_row_major

function change_input_to_row_major(x)
    return permutedims(reshape(x, reverse(size(x))), length(size(x)):-1:1)
end

function change_output_to_row_major(x)
    return reshape(permutedims(x, length(size(x)):-1:1), size(x))
end