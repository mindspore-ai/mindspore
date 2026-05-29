MS_LOG(DEBUG) << op_name() << " call start";
${value_tuple_convert}
${const_number_convert}

ProfileTrackerTask();
auto __result = ${helper_name}(${real_call_args});

ProfileTrackerInput(${call_args});
ProfileTrackerOutput(__result);
MS_LOG(DEBUG) << op_name() << " call end. output size: " << ${output_size};
return __result;
