PyObject* ${func_name}_Base(const PrimitivePtr &prim, PyObject* args) {
#ifndef ENABLE_TEST
  ${mark_side_effect}
  static pynative::Converter converter(&ops::${op_def_name});
  converter.Parse(args);
  ${parser_body}
  if (converter.has_fallback()) {
    auto op_call = std::make_shared<OpCall>("${class_name}", [prim](const py::args &args, const py::kwargs &kwargs){
      auto list_args = py::list(args);
      return py::reinterpret_steal<py::object>(${func_name}_Base(prim, list_args.ptr()));
    });
    return pynative::HandleFallback(args, py::cast(op_call));
  }
  return ${func_name}_OP(prim, converter.source_type(), ${op_args});
#else
  py::object py_args = py::reinterpret_borrow<py::object>(args);
  py::object res = PyNativeAlgo::PyBoost::RunPyFunction(prim, py_args);
  return res.release().ptr();
#endif
}

