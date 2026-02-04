class ${cpp_func_name}Functional : public Functional {
 public:
  ${cpp_func_name}Functional() : Functional("${func_name}") {};
  ~${cpp_func_name}Functional() = default;
  py::object Call(const py::args &args, const py::kwargs &kwargs) {
    static mindspore::pynative::PythonArgParser parser({
      ${signatures}
      }, "${func_name}");
    auto parse_args = parser.Parse(args.ptr(), kwargs.ptr(), false);
#ifndef ENABLE_TEST
    auto backend = DeviceManagerConf::GetInstance()->device_type();
    static auto callback = [this](const py::args &args, const py::kwargs &kwargs){
      return py::reinterpret_steal<py::object>(Call(args, kwargs));
    };
    switch (parse_args.GetOvertLoadIndex()) {
      ${dispatch_cases}
    }
    return py::none();
#else
    ${ut_overload_body}
#endif
  }
};

static std::shared_ptr<${cpp_func_name}Functional> ${func_name}_instance = std::make_shared<${cpp_func_name}Functional>();

