${return_type} ${op_name}(${input_args_with_type}) {
  MS_LOG(DEBUG) << "In ${op_name} function";

  bool skip_tracker = ProfileTracker::ProfileTrackerTask(prim::kPrim${class_name});

  // ${op_name}_impl is customized, which inherently includes differentiation and
  // is implemented in mindspore/ccsrc/pynative/utils/pyboost/functions/customize/view_impl.h
  auto output = ${op_name}_impl(${input_args});

  ProfileTracker::ProfileTrackerInput(prim::kPrim${class_name}, skip_tracker, ${input_args});
  ProfileTracker::ProfileTrackerOutput(prim::kPrim${class_name}, skip_tracker, output);

  MS_LOG(DEBUG) << "Out ${op_name} function";
  return output;
}
