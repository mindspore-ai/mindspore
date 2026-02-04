#include "module.h"

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.doc() = "A custom module for operators";
  ModuleRegistry::Instance().RegisterAll(m);
}
