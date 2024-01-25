void ${func_name}(const PrimitivePtr &primitive, const std::string& device_target,
                       const std::vector<ValuePtr> &inputs, VectorRef *op_outputs) {
  // Create op
  auto op = CREATE_PYBOOST_OP(${op_name}, device_target);
  op->set_primitive(primitive);

  // Run op
  ${convert_body}
  (void)op->Call(${call_args});
  MS_EXCEPTION_IF_NULL(op_outputs);
  (void)std::transform(op->outputs().begin(), op->outputs().end(), std::back_inserter(*op_outputs),
                       [] (const auto &item) {return item;});
}
