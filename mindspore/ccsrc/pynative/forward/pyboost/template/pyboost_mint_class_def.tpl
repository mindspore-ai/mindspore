class ${class_name}Functional : public Functional {
 public:
  ${class_name}Functional() : Functional("${func_name}") {};
  ~${class_name}Functional() = default;
  py::object Call(const py::args &args, const py::kwargs &kwargs) {
    static mindspore::pynative::PythonArgParser parser({
      ${signatures}
    }, "${func_name}");
    auto parse_args = parser.Parse(args.ptr(), kwargs.ptr(), false);
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    auto backend = DeviceManagerConf::GetInstance()->device_type();
    static auto callback = [this](const py::args &args, const py::kwargs &kwargs){
      return py::reinterpret_steal<py::object>(Call(args, kwargs));
    };
    ${device_dispatcher}
    return py::none();
  }
};

static std::shared_ptr<${class_name}Functional> ${func_name}_instance = std::make_shared<${class_name}Functional>();

