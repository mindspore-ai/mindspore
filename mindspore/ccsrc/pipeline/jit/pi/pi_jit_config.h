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
#ifndef MINDSPORE_PI_JIT_CONFIG_H
#define MINDSPORE_PI_JIT_CONFIG_H

#include <set>
#include <string>
#include "pybind11/pybind11.h"

namespace mindspore {
namespace pijit {
namespace py = pybind11;

class GraphJitConfig {
 public:
  enum Options {
    kBoolConf = 0,
    kAutoJitCell,
    kAutoJit,
    kReplaceNNCellByConstruct,
    kPrintAfterAll,
    kPrintTraceback,
    kPrintBB,
    kPrintCFG,
    kInterpretCapturedCode,
    kCompileWithoutCapture,
    kCompileWithTry,
    kGuardSpecializeScalar,
    kGuardSpecializeContainer,
    kGuardSpecializeTensor,
    kGuardDetachObject,
    kPrintGuard,
    kReuseGraph,
    kPrintReuseGraph,
    kAutoCleanCache,
    kPruneCase,
    kLoopUnrolling,
    kInferOnly,
    kInferPrimitive,
    kStrictTrace,
    kPerfStatistics,
    kLogGraphBreak,
    kLogPerf,
    kLogGuardPerf,
    kTestGraphIR,
    kEnableOptimizeForAttrItem,
    kEnableEliminateUnusedOperation,
    kEnableGeneratorExpressionToTuple,
    kFeatureBreakAtInlinedFunction,
    kEnableDynamicShape,
    kTraceFlag,
    kSkipException,
    /* ------------------------------ */
    kIntConf,
    kMaxInlineDepth,
    kMaxTraceDepth,
    kMaxPruneCase,
    kMaxLoopUnrolling,
    kStaticGraphBytecodeMin,
    kPerfStatisticsCount,
    kPerfStatisticsScale10000x,
    kInferPrimitiveMask,
    kInferPrimitiveMax,
    kLimitGraphSize,
    kLimitGraphCount,
    /* ------------------------------ */
    kStrListConf,
    kAllowedInlineModules,
    kPSJitStrictCells,
    kJitForbidden,
    kOptionsCount
  };
  GraphJitConfig();
  explicit GraphJitConfig(const py::object &c);
  bool GetBoolConfig(Options o) const { return o > kBoolConf && o < kIntConf ? bool_conf[o - kBoolConf] : false; }
  int getIntConfig(Options o) const { return o > kIntConf && o < kStrListConf ? int_conf[o - kIntConf] : 0; }
  const auto *getSetConfig(Options o) const {
    return o > kStrListConf && o < kOptionsCount ? &set_conf[o - kStrListConf] : nullptr;
  }

  bool ShouldAutoJit(PyFrameObject *f);
  bool CheckJitForbidden(const py::object &callable);
  bool CheckJitConstexpr(const py::object &code);

  void AddAllowedInlineModules(const std::string &module_name);
  void AddPSJitStrictCells(const std::string &type_str);

  bool AddJitConstexpr(PyObject *list);
  bool AddJitForbidden(PyObject *callable_list);
  bool AddAllowedInlineModules(PyObject *list);
  bool AddPSJitStrictCells(PyObject *list);
  bool SetAutoJitFilter(PyObject *callable);

  template <Options o>
  bool SetBool(PyObject *value) {
    static_assert(o > kBoolConf && o < kIntConf);
    bool_conf[o - kBoolConf] = value == Py_True;
    return true;
  }

  template <Options o>
  bool SetInt(PyObject *value) {
    static_assert(o > kIntConf && o < kStrListConf);
    int res = PyLong_AsLong(value);
    if (PyErr_Occurred()) {
      PyErr_Clear();
      return false;
    }
    int_conf[o - kIntConf] = res;
    return true;
  }

  static void ApplyAutoJitCell();

 private:
  bool bool_conf[kIntConf - kBoolConf];
  int int_conf[kStrListConf - kIntConf];
  std::set<std::string> set_conf[kOptionsCount - kStrListConf];
};

extern GraphJitConfig kPIJitConfigDefault;

}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_CONFIG_H
