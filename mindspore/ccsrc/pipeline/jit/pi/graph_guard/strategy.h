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
#ifndef MINDSPORE_PI_JIT_STRATEGY_H
#define MINDSPORE_PI_JIT_STRATEGY_H

#include <vector>
#include "pybind11/pybind11.h"
#include "pipeline/jit/pi/graph_guard/perf.h"
#include "pipeline/jit/pi/graph_guard/cache.h"
#include "mindapi/base/shape_vector.h"

using PyObjectArray = std::vector<PyObject *>;

namespace mindspore {
namespace pijit {

class OptStrategy {
 public:
  typedef enum {
    kExecPyNative = 0,
    kExecGraph,
    kExecCount,
  } ExecKind;
  static ExecKind MakeExecStrategyByPerf(OptPerfPtr graph_perf, OptPerfPtr pynative_perf, int count,
                                         double adj_coef = 0.1);
  static ExecKind MakeExecStrategyByComplex(PyCodeObject *code, int threshold);
  static void MakeGCStrategy(OptCodeHubPtr hub, int limit_size, int limit_count, bool enable_dynamicshape,
                             OptCodePtr except);
  typedef enum {
    kCalcUnsupported = 0,
    kCalcShape,
    kCalcValue,
    kCalcCount,
  } CalcKind;
  static CalcKind MakeCalcStrategyByInputs(int bytecode, int opargs, const PyObjectArray &objs);
  static CalcKind MakeCalcStrategyByShape(const ShapeVector &shape);
  static OptCodeSet MakeGuardListStrategyByFrame(const PyFrameObject *frame, const OptCodeSet &codes);
  static GuardItemVector MakeGuardItemListStrategyByFrame(const PyFrameObject *frame, const GuardItemVector &list);
};

}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_STRATEGY_H
