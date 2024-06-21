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
    kAutoGrad,
    kAutoJit,
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
    kEnableMsApiInfer,
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
    kGuardRelaxCount,
    /* ------------------------------ */
    kOptionsCount
  };
  GraphJitConfig();
  explicit GraphJitConfig(const py::object &c);
  bool GetBoolConfig(Options o) const { return o > kBoolConf && o < kIntConf ? bool_conf[o - kBoolConf] : false; }
  int getIntConfig(Options o) const { return o > kIntConf && o < kOptionsCount ? int_conf[o - kIntConf] : 0; }
  const auto &allowed_inline_modules() const { return allowed_inline_modules_; }

  bool ShouldAutoJit(PyFrameObject *f);

  void AddAllowedInlineModules(const std::string &module_name);

  bool SetAutoJitFilter(PyObject *callable);
  bool AddJitRelaxGuard(PyObject *list);
  bool AddJitConstexpr(PyObject *callable_list);
  bool AddJitForbidden(PyObject *callable_list);
  bool AddAllowedInlineModules(PyObject *str_list);
  std::string getJitLevel() const;
  bool AddJitLevel(PyObject *str);

  template <Options o>
  bool SetBool(PyObject *value) {
    static_assert(o > kBoolConf && o < kIntConf);
    bool_conf[o - kBoolConf] = value == Py_True;
    return true;
  }

  template <Options o>
  bool SetInt(PyObject *value) {
    static_assert(o > kIntConf && o < kOptionsCount);
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
  std::set<std::string> allowed_inline_modules_;
  int int_conf[kOptionsCount - kIntConf];
  bool bool_conf[kIntConf - kBoolConf];
  std::string jit_level;
};

extern GraphJitConfig kPIJitConfigDefault;

}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_CONFIG_H
