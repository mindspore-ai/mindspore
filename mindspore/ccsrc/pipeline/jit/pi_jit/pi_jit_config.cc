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

namespace mindspore {
namespace jit {
namespace graph {

GraphJitConfig kPIJitConfigDefault;

static const std::unordered_map<std::string, GraphJitConfig::Options> bool_key_map = {
  {"replace_nncell_by_construct", GraphJitConfig::kReplaceNNCellByConstruct},
  {"capture_msadapter_forward", GraphJitConfig::kCapturedMSadapterForward},
  {"print_after_all", GraphJitConfig::kPrintAfterAll},
  {"print_tb", GraphJitConfig::kPrintTraceback},
  {"print_bb", GraphJitConfig::kPrintBB},
  {"print_cfg", GraphJitConfig::kPrintCFG},
  {"interpret_captured_code", GraphJitConfig::kInterpretCapturedCode},
  {"compile_without_capture", GraphJitConfig::kCompileWithoutCapture},
  {"compile_with_try", GraphJitConfig::kCompileWithTry},
  {"enable_guard", GraphJitConfig::kEnableGuard},
  {"guard_subroutine", GraphJitConfig::kGuardSubRoutine},
  {"specialize_scalar", GraphJitConfig::kGuardSpecializeScalar},
  {"specialize_container", GraphJitConfig::kGuardSpecializeContainer},
  {"specialize_tensor", GraphJitConfig::kGuardSpecializeTensor},
  {"print_guard", GraphJitConfig::kPrintGuard},
  {"auto_clean_cache", GraphJitConfig::kAutoCleanCache},
  {"prune_case", GraphJitConfig::kPruneCase},
  {"loop_unrolling", GraphJitConfig::kLoopUnrolling},
  {"infer_primitive", GraphJitConfig::kInferPrimitive},
  {"strict_trace", GraphJitConfig::kStrictTrace},
  {"perf_statistics", GraphJitConfig::kPerfStatistics},
  {"LOG_GRAPH_BREAK", GraphJitConfig::kLogGraphBreak}
  // kEnableOptimizeForAttrItem
  // kEnableEliminateUnusedOperation
};

static const std::unordered_map<std::string, GraphJitConfig::Options> int_key_map = {
  {"MAX_INLINE_DEPTH", GraphJitConfig::kMaxInlineDepth},
  {"MAX_PRUNE_CASE", GraphJitConfig::kMaxPruneCase},
  {"MAX_LOOP_UNROLLING", GraphJitConfig::kMaxLoopUnrolling},
  {"INFER_PRIMITIVE_MASK", GraphJitConfig::kInferPrimitiveMask},
  {"INFER_PRIMITIVE_MAX", GraphJitConfig::kInferPrimitiveMax},
  {"STATIC_GRAPH_BYTECODE_MIN", GraphJitConfig::kStaticGraphBytecodeMin},
  {"PERF_STATISTICS_SCALE_10000X", GraphJitConfig::kPerfStatisticsScale10000x},
};

GraphJitConfig::GraphJitConfig() {
  bool_conf[kReplaceNNCellByConstruct - kBoolConf] = false;
  bool_conf[kCapturedMSadapterForward - kBoolConf] = true;
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
}

bool GraphJitConfig::SetBool(const char *k, PyObject *value) {
  auto b_iter = bool_key_map.find(k);
  if (b_iter == bool_key_map.end()) {
    return false;
  }
  bool_conf[b_iter->second - kBoolConf] = value == Py_True;
  return true;
}

bool GraphJitConfig::SetInt(const char *k, PyObject *value) {
  auto i_iter = int_key_map.find(k);
  if (i_iter == int_key_map.end()) {
    return false;
  }
  int res = PyLong_AsLong(value);
  if (PyErr_Occurred()) {
    PyErr_Clear();
    return false;
  }
  int_conf[i_iter->second - kIntConf] = res;
  return true;
}

void GraphJitConfig::AddAllowedInlineModules(PyObject *list) {
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
}

void GraphJitConfig::AddAllowedInlineModules(const std::string &module_name) {
  set_conf[kAllowedInlineModules - kStrListConf].insert(module_name);
}

GraphJitConfig::GraphJitConfig(const py::object &c) {
  *this = kPIJitConfigDefault;
  (void)c.cast<py::dict>();
  PyObject *key, *value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(c.ptr(), &pos, &key, &value)) {
    if (PyUnicode_Check(key)) {
      const char *k = PyUnicode_AsUTF8(key);
      if (SetBool(k, value)) {
        continue;
      }
      if (SetInt(k, value)) {
        continue;
      }
      if (!strcmp(k, "allowed_inline_modules")) {
        AddAllowedInlineModules(value);
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
