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
#ifndef MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_STRATEGY_H
#define MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_STRATEGY_H

#include <Python.h>
#include <vector>
#include "pipeline/jit/pi/graph_guard/perf.h"
#include "pipeline/jit/pi/graph_guard/cache.h"

using PyObjectArray = std::vector<PyObject *>;

namespace mindspore {
namespace jit {
namespace graph {

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
  static void MakeGCStrategy(OptCodeHubPtr hub, int limit_size, int limit_count, OptCodePtr except);
  typedef enum {
    kCalcUnsupported = 0,
    kCalcShape,
    kCalcValue,
    kCalcCount,
  } CalcKind;
  static CalcKind MakeCalcStrategyByInputs(int bytecode, int opargs, const PyObjectArray &objs);
};

}  // namespace graph
}  // namespace jit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_STRATEGY_H
