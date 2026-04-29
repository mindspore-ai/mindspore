MS_LOG(DEBUG) << op_name() << " call start";
InferOutput(${call_args});
// ValueTuple to std::vector
${value_tuple_convert}
// Convert ValuePtr to c++ scalar
${const_number_convert}

auto op = get_op();
${create_input_address}
${inplace_process}
TensorAllocate(op, outputs_);

ProfileTrackerTask();
const auto &outputs = op->outputs();

DISPATCH_LAUNCH_ACLNN(${aclnn_name}, ${real_call_args}${outputs}${cube_math_type});

ProfileTrackerInput(${call_args});
ProfileTrackerOutput(${return_values});
MS_LOG(DEBUG) << op_name() << " call end. output size: " << outputs.size();
return ${return_values};
