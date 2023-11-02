/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "pipeline/jit/pi_jit/pi_jit_config.h"
#include <string>
#include <unordered_map>
#include "utils/log_adapter.h"
#include "pipeline/jit/pi_jit/external.h"
#include "pipeline/jit/pi_jit/utils/utils.h"
#include "pipeline/jit/pi_jit/pydef.h"

namespace mindspore {
namespace jit {
namespace graph {

GraphJitConfig kPIJitConfigDefault;

static const std::unordered_map<std::string, bool (GraphJitConfig::*)(PyObject *)> key_map = {
  // debug option, until fix the error of copied function reuse, if compiled results recursive call self
  {"copy_once_if_trace_break", &GraphJitConfig::SetBool<GraphJitConfig::kCopyFuncOnlyOnceIfTraceBreak>},
  {"auto_jit_func_filter", &GraphJitConfig::SetAutoJitFilter},
  // remove this config if 'strict_mode_cells' works well, and default inline all construct
  {"replace_nncell_by_construct", &GraphJitConfig::SetBool<GraphJitConfig::kReplaceNNCellByConstruct>},
  {"print_after_all", &GraphJitConfig::SetBool<GraphJitConfig::kPrintAfterAll>},
  {"print_tb", &GraphJitConfig::SetBool<GraphJitConfig::kPrintTraceback>},
  {"print_bb", &GraphJitConfig::SetBool<GraphJitConfig::kPrintBB>},
  {"print_cfg", &GraphJitConfig::SetBool<GraphJitConfig::kPrintCFG>},
  {"interpret_captured_code", &GraphJitConfig::SetBool<GraphJitConfig::kInterpretCapturedCode>},
  {"compile_without_capture", &GraphJitConfig::SetBool<GraphJitConfig::kCompileWithoutCapture>},
  {"compile_with_try", &GraphJitConfig::SetBool<GraphJitConfig::kCompileWithTry>},
  {"enable_guard", &GraphJitConfig::SetBool<GraphJitConfig::kEnableGuard>},
  {"guard_subroutine", &GraphJitConfig::SetBool<GraphJitConfig::kGuardSubRoutine>},
  {"specialize_scalar", &GraphJitConfig::SetBool<GraphJitConfig::kGuardSpecializeScalar>},
  {"specialize_container", &GraphJitConfig::SetBool<GraphJitConfig::kGuardSpecializeContainer>},
  {"specialize_tensor", &GraphJitConfig::SetBool<GraphJitConfig::kGuardSpecializeTensor>},
  {"print_guard", &GraphJitConfig::SetBool<GraphJitConfig::kPrintGuard>},
  {"auto_clean_cache", &GraphJitConfig::SetBool<GraphJitConfig::kAutoCleanCache>},
  {"prune_case", &GraphJitConfig::SetBool<GraphJitConfig::kPruneCase>},
  {"loop_unrolling", &GraphJitConfig::SetBool<GraphJitConfig::kLoopUnrolling>},
  {"infer_primitive", &GraphJitConfig::SetBool<GraphJitConfig::kInferPrimitive>},
  {"strict_trace", &GraphJitConfig::SetBool<GraphJitConfig::kStrictTrace>},
  {"perf_statistics", &GraphJitConfig::SetBool<GraphJitConfig::kPerfStatistics>},
  {"LOG_GRAPH_BREAK", &GraphJitConfig::SetBool<GraphJitConfig::kLogGraphBreak>},
  // kEnableOptimizeForAttrItem
  // kEnableEliminateUnusedOperation
  {"MAX_INLINE_DEPTH", &GraphJitConfig::SetInt<GraphJitConfig::kMaxInlineDepth>},
  {"MAX_PRUNE_CASE", &GraphJitConfig::SetInt<GraphJitConfig::kMaxPruneCase>},
  {"MAX_LOOP_UNROLLING", &GraphJitConfig::SetInt<GraphJitConfig::kMaxLoopUnrolling>},
  {"INFER_PRIMITIVE_MASK", &GraphJitConfig::SetInt<GraphJitConfig::kInferPrimitiveMask>},
  {"INFER_PRIMITIVE_MAX", &GraphJitConfig::SetInt<GraphJitConfig::kInferPrimitiveMax>},
  {"STATIC_GRAPH_BYTECODE_MIN", &GraphJitConfig::SetInt<GraphJitConfig::kStaticGraphBytecodeMin>},
  {"PERF_STATISTICS_SCALE_10000X", &GraphJitConfig::SetInt<GraphJitConfig::kPerfStatisticsScale10000x>},
  {"allowed_inline_modules", &GraphJitConfig::AddAllowedInlineModules},
  {"strict_mode_cells", &GraphJitConfig::AddPSJitStrictCells},
  {"pijit_forbidden", &GraphJitConfig::AddJitForbidden},
};

GraphJitConfig::GraphJitConfig() {
  bool_conf[kCopyFuncOnlyOnceIfTraceBreak - kBoolConf] = false;
  bool_conf[kAutoJit - kBoolConf] = false;
  bool_conf[kReplaceNNCellByConstruct - kBoolConf] = false;
  bool_conf[kPrintAfterAll - kBoolConf] = false;
  bool_conf[kPrintTraceback - kBoolConf] = false;
  bool_conf[kPrintBB - kBoolConf] = false;
  bool_conf[kPrintCFG - kBoolConf] = false;
  bool_conf[kInterpretCapturedCode - kBoolConf] = false;
  bool_conf[kCompileWithoutCapture - kBoolConf] = false;
  bool_conf[kCompileWithTry - kBoolConf] = false;
  bool_conf[kEnableGuard - kBoolConf] = true;
  bool_conf[kGuardSubRoutine - kBoolConf] = false;
  bool_conf[kGuardSpecializeScalar - kBoolConf] = true;
  bool_conf[kGuardSpecializeContainer - kBoolConf] = false;
  bool_conf[kGuardSpecializeTensor - kBoolConf] = false;
  bool_conf[kPrintGuard - kBoolConf] = false;
  bool_conf[kAutoCleanCache - kBoolConf] = false;
  bool_conf[kPruneCase - kBoolConf] = true;
  bool_conf[kLoopUnrolling - kBoolConf] = true;
  bool_conf[kInferPrimitive - kBoolConf] = true;
  bool_conf[kStrictTrace - kBoolConf] = true;
  bool_conf[kPerfStatistics - kBoolConf] = false;
  bool_conf[kLogGraphBreak - kBoolConf] = false;

  /*'EnableOptimizeForAttrItem' options must be ensure that multiple calls of the
   *__getattr__, __getitem__ function of the user-defined object do not affect the correctness.
   */
  bool_conf[kEnableOptimizeForAttrItem - kBoolConf] = true;
  bool_conf[kEnableEliminateUnusedOperation - kBoolConf] = false;

  int_conf[kMaxInlineDepth - kIntConf] = 8;
  int_conf[kMaxPruneCase - kIntConf] = -1;
  int_conf[kMaxLoopUnrolling - kIntConf] = 100;
  int_conf[kInferPrimitiveMask - kIntConf] = 7;
  int_conf[kInferPrimitiveMax - kIntConf] = 0;
  int_conf[kStaticGraphBytecodeMin - kIntConf] = 0;
  int_conf[kPerfStatisticsScale10000x - kIntConf] = 1000;

  set_conf[kAllowedInlineModules - kStrListConf] = {"mindspore"};
  set_conf[kPSJitStrictCells - kStrListConf] = {};
}

bool GraphJitConfig::AddAllowedInlineModules(PyObject *list) {
  py::object l = py::reinterpret_borrow<py::object>(list);
  for (const auto &i : py::iter(l)) {
    const char *name = nullptr;
    if (PyUnicode_Check(i.ptr())) {
      name = PyUnicode_AsUTF8(i.ptr());
    } else if (PyModule_Check(i.ptr())) {
      name = PyModule_GetName(i.ptr());
    } else {
      continue;
    }
    if (name == nullptr) {
      PyErr_Clear();
      continue;
    }
    AddAllowedInlineModules(name);
  }
  return true;
}

void GraphJitConfig::AddAllowedInlineModules(const std::string &module_name) {
  set_conf[kAllowedInlineModules - kStrListConf].insert(module_name);
}

void GraphJitConfig::AddPSJitStrictCells(const std::string &type_str) {
  set_conf[kPSJitStrictCells - kStrListConf].insert(type_str);
}

bool GraphJitConfig::AddPSJitStrictCells(PyObject *list) {
  py::object l = py::reinterpret_borrow<py::object>(list);
  py::object func = Utils::GetModuleAttr("mindspore.nn", "Cell", false, false);
  for (const auto &i : py::iter(l)) {
    if (py::isinstance(i, func)) {
      AddPSJitStrictCells(std::string(py::str(reinterpret_cast<PyObject *>(Py_TYPE(i.ptr())))));
      continue;
    }
    if (PyObject_IsSubclass(i.ptr(), func.ptr()) == true) {
      AddPSJitStrictCells(std::string(py::str(i.ptr())));
      continue;
    }
    MS_LOG(WARNING) << "for config option 'strict_mode_cells' all elements must be subclass of mindspore.nn.Cell";
    return false;
  }
  return true;
}

bool GraphJitConfig::SetAutoJitFilter(PyObject *callable) {
  if (!PyCallable_Check(callable)) {
    MS_LOG(WARNING) << "PIJit option 'auto_jit_func_filter' only accept callable, but got "
                    << std::string(py::str(callable));
    return false;
  }
  (void)SetBool<kAutoJit>(Py_True);
  py::object func = Utils::GetModuleAttr("mindspore", "jit", false, true);
  func.attr("__auto_pijit_filter__") = callable;
  return true;
}

bool GraphJitConfig::ShouldAutoJit(PyFrameObject *f) {
  if (!GetBoolConfig(kAutoJit)) {
    return false;
  }
  py::object func = Utils::GetModuleAttr("mindspore", "jit", false, false);
  if (func.ptr() == nullptr) {
    // mindspore module is unload
    (void)SetBool<kAutoJit>(Py_False);
    return false;
  }
  py::object filter = func.attr("__auto_pijit_filter__");
  PyObject *arg = reinterpret_cast<PyObject *>(f);
  PyObject *res = PyObject_Vectorcall(filter.ptr(), &arg, 1, nullptr);
  if (PyErr_Occurred()) {
    MS_LOG(ERROR) << "***" << py::error_already_set().what() << "*** at " << std::string(py::str(filter)) << " ignored";
    PyErr_Clear();
    (void)SetBool<kAutoJit>(Py_False);
    return false;
  }
  Py_DECREF(res);
  return res == Py_True;
}

static std::string GetCodeKey(PyCodeObject *co) {
  std::stringstream s;
  s << co << PyUnicode_AsUTF8(co->co_name);
  return s.str();
}

bool GraphJitConfig::AddJitForbidden(PyObject *list) {
  for (const py::handle &i : py::iter(list)) {
    py::object code = GetPyCodeObject(py::cast<py::object>(i));
    PyCodeObject *co = reinterpret_cast<PyCodeObject *>(code.ptr());
    if (co == nullptr) {
      MS_LOG(WARNING) << "config options 'jit_forbidden', can't find the code of " << std::string(py::str(i));
      return false;
    }
    set_conf[kJitForbidden - kStrListConf].insert(GetCodeKey(co));
  }
  return true;
}

bool GraphJitConfig::CheckJitForbidden(const py::object &code) {
  py::object h = GetPyCodeObject(code);
  PyCodeObject *co = reinterpret_cast<PyCodeObject *>(h.ptr());
  if (co == nullptr) {
    return false;
  }
  const auto &s = set_conf[kJitForbidden - kStrListConf];
  return s.find(GetCodeKey(co)) != s.end();
}

GraphJitConfig::GraphJitConfig(const py::object &c) {
  *this = kPIJitConfigDefault;
  (void)c.cast<py::dict>();
  PyObject *key, *value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(c.ptr(), &pos, &key, &value)) {
    if (PyUnicode_Check(key)) {
      const char *k = PyUnicode_AsUTF8(key);
      auto iter = key_map.find(k);
      if (iter != key_map.end() && (this->*(iter->second))(value)) {
        continue;
      }
    }
    MS_LOG(WARNING) << "unknown PIJit options: " << std::string(py::str(key)) << ":" << std::string(py::str(value));
  }
}

}  // namespace graph
}  // namespace jit

void update_pijit_default_config(const py::kwargs &conf) {
  mindspore::jit::graph::kPIJitConfigDefault = mindspore::jit::graph::GraphJitConfig(conf);
}

}  // namespace mindspore
