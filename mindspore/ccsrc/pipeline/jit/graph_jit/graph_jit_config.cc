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
#include "pipeline/jit/graph_jit/graph_jit_config.h"
#include <string>
#include "utils/log_adapter.h"

namespace mindspore {
namespace jit {
namespace graph {

static py::object GetConfig(const py::object &dict, const std::string &key) {
  PyObject *value = _PyDict_Pop(dict.ptr(), py::str(key).ptr(), Py_None);
  return py::reinterpret_steal<py::object>(value);
}

static bool CheckConfigBoolValue(const py::object &c, const std::string &key, bool default_value = false) {
  py::object o = GetConfig(c, key);
  PyObject *value = o.ptr();
  if (value == Py_None) {
    return default_value;
  }
  return value == Py_True;
}

static int CheckConfigIntValue(const py::object &c, const std::string &key, int default_value = 0) {
  py::object o = GetConfig(c, key);
  PyObject *value = o.ptr();
  if (value == Py_None) {
    return default_value;
  }
  int res = PyLong_AsLong(value);
  if (PyErr_Occurred()) {
    PyErr_Clear();
    return default_value;
  }
  return res;
}

static std::set<std::string> GetAllowedInlineModules(const py::object &c) {
  std::set<std::string> modules = {"mindspore"};
  py::object l = GetConfig(c, "allowed_inline_modules");
  if (l.ptr() == Py_None) {
    return modules;
  }
  for (const auto &i : py::iter(l)) {
    if (PyUnicode_Check(i.ptr())) {
      modules.insert(PyUnicode_AsUTF8(i.ptr()));
    }
  }
  return modules;
}

GraphJitConfig::GraphJitConfig(const py::object &c) {
  bool_conf[kReplaceNNCellByConstruct - kBoolConf] = CheckConfigBoolValue(c, "replace_nncell_by_construct");
  bool_conf[kPrintAfterAll - kBoolConf] = CheckConfigBoolValue(c, "print_after_all");
  bool_conf[kPrintTraceback - kBoolConf] = CheckConfigBoolValue(c, "print_tb");
  bool_conf[kPrintBB - kBoolConf] = CheckConfigBoolValue(c, "print_bb");
  bool_conf[kPrintCFG - kBoolConf] = CheckConfigBoolValue(c, "print_cfg");
  bool_conf[kPrintLastFrameIfBreakGraph - kBoolConf] = CheckConfigBoolValue(c, "print_last_frame_if_break_graph");
  bool_conf[kInterpretCapturedCode - kBoolConf] = CheckConfigBoolValue(c, "interpret_captured_code");
  bool_conf[kCompileWithoutCapture - kBoolConf] = CheckConfigBoolValue(c, "compile_without_capture");
  bool_conf[kNotInlineAnyFunction - kBoolConf] = CheckConfigBoolValue(c, "not_inline_any_function");
  bool_conf[kDebugGraphBreakAtUnsupportedOperations - kBoolConf] =
    CheckConfigBoolValue(c, "graph_break_at_unsupported_operations", true);
  bool_conf[kEnableGuard - kBoolConf] = CheckConfigBoolValue(c, "enable_guard", true);
  bool_conf[kGuardSpecializeIntFloat - kBoolConf] = CheckConfigBoolValue(c, "specialize_int_float", true);
  bool_conf[kGuardSpecializeTensor - kBoolConf] = CheckConfigBoolValue(c, "specialize_tensor");
  bool_conf[kGuardSubRoutine - kBoolConf] = CheckConfigBoolValue(c, "guard_subroutine");
  bool_conf[kPrintGuard - kBoolConf] = CheckConfigBoolValue(c, "print_guard");
  bool_conf[kAutoCleanCache - kBoolConf] = CheckConfigBoolValue(c, "auto_clean_cache");
  bool_conf[kPruneCase - kBoolConf] = CheckConfigBoolValue(c, "prune_case", true);
  bool_conf[kLoopUnrolling - kBoolConf] = CheckConfigBoolValue(c, "loop_unrolling", true);
  bool_conf[kInferPrimitive - kBoolConf] = CheckConfigBoolValue(c, "infer_primitive", true);

  /*'EnableOptimizeForAttrItem' options must be ensure that multiple calls of the
   *__getattr__, __getitem__ function of the user-defined object do not affect the correctness.
   */
  bool_conf[kEnableOptimizeForAttrItem - kBoolConf] = CheckConfigBoolValue(c, "EnableOptimizeForAttrItem", true);

  int inline_depth = CheckConfigIntValue(c, "MAX_INLINE_DEPTH", 8);
  int_conf[kMaxInlineDepth - kIntConf] = inline_depth < 0 ? 8 : inline_depth;
  int_conf[kMaxPruneCase - kIntConf] = CheckConfigIntValue(c, "MAX_PRUNE_CASE", -1);
  int_conf[kMaxLoopUnrolling - kIntConf] = CheckConfigIntValue(c, "MAX_LOOP_UNROLLING", 100);
  int_conf[kInferPrimitiveMask - kIntConf] = CheckConfigIntValue(c, "INFER_PRIMITIVE_MASK", 7);
  int_conf[kInferPrimitiveMax - kIntConf] = CheckConfigIntValue(c, "INFER_PRIMITIVE_MAX", 0);

  set_conf[kAllowedInlineModules - kStrListConf] = GetAllowedInlineModules(c);
  if (PyDict_Size(c.ptr()) > 0) {
    MS_LOG(WARNING) << "unknown PIJit jit_config options: " << std::string(py::str(c));
  }
}
}  // namespace graph
}  // namespace jit
}  // namespace mindspore
