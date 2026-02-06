/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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
#include "frontend/jit/pi/graph_guard/strategy.h"
#include <algorithm>
#include <limits>
#include <string>
#include <vector>
#include <map>
#include <set>
#include "pybind11/pybind11.h"
#include "include/frontend/operator/primitive_py.h"
#include "include/utils/convert_utils_py.h"
#include "include/frontend/jit/ps/executor/jit_executor_py.h"
#include "frontend/jit/pi/utils/utils.h"
#include "frontend/jit/pi/python_adapter/pydef.h"
#include "frontend/jit/pi/utils/opcode_declare.h"
#include "include/utils/tensor_py.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace pijit {

OptStrategy::ExecKind OptStrategy::MakeExecStrategyByPerf(OptPerfPtr graph_perf, OptPerfPtr pynative_perf, int count,
                                                          double adj_coef) {
  PerfStatisticsPtr graph_stat = graph_perf->GetStatistics();
  PerfStatisticsPtr pynative_stat = graph_perf->GetStatistics();
  if (graph_stat->GetTotalCount() < count) {
    return ExecKind::kExecGraph;
  } else if (pynative_stat->GetTotalCount() < count) {
    return ExecKind::kExecPyNative;
  } else {
    if (graph_stat->GetAverageDuration() * (1 + adj_coef) > pynative_stat->GetAverageDuration()) {
      return ExecKind::kExecPyNative;
    } else {
      return ExecKind::kExecGraph;
    }
  }
}

OptStrategy::ExecKind OptStrategy::MakeExecStrategyByComplex(PyCodeObject *co, int threshold) {
  // currently just use instruction count to judge whether to use graph build
  // later it need cost model to make judgement here
  if (co != nullptr && _PyCode_NBYTES(co) < threshold) {
    return ExecKind::kExecPyNative;
  } else {
    return ExecKind::kExecGraph;
  }
}

static bool CompareOptCodeByCount(OptCodePtr a, OptCodePtr b) {
  if (a->Count() > b->Count()) {
    return true;
  } else {
    return false;
  }
}

static constexpr int64_t kDynamicShapeLimitCount = 3;

void OptStrategy::MakeGCStrategy(OptCodeHubPtr hub, int limit_size, int limit_count, bool enable_dynamicshape,
                                 OptCodePtr except) {
  if (limit_size <= 0 && limit_count <= 0) {
    if (!enable_dynamicshape) {
      return;
    }
    limit_count = kDynamicShapeLimitCount;
  }
  std::vector<OptCodeSet> vec = hub->GetAllOptTarget();
  for (auto set : vec) {
    std::sort(set.begin(), set.end(), CompareOptCodeByCount);
    auto it = std::find(set.begin(), set.end(), except);
    if (it != set.end()) {
      set.erase(it);
    }
    if (limit_count > 0) {
      if (set.size() > (size_t)limit_count) {
        OptCodeSet toDel;
        toDel.insert(toDel.begin(), set.begin() + limit_count, set.end());
        for (auto item : toDel) {
          hub->DelOptTarget(item);
        }
      }
    }
    if (limit_size > 0) {
      auto graph_executor = pipeline::GetExecutor();
      OptCodeSet toDel;
      for (auto item : set) {
        if (limit_size == 0) {
          toDel.push_back(item);
        }
        std::string phase = item->GetPhase();
        if (phase.size() > 0) {
          FuncGraphPtr ms_func_graph = graph_executor->GetFuncGraph(phase);
          MS_EXCEPTION_IF_NULL(ms_func_graph);
          int node_count = SizeToInt(ms_func_graph->nodes().size());
          for (auto fg : ms_func_graph->func_graphs_used_total()) {
            node_count += SizeToInt(fg->nodes().size());
          }
          if (limit_size > node_count) {
            limit_size -= node_count;
          } else {
            limit_size = 0;
          }
        }
      }
      for (auto item : toDel) {
        hub->DelOptTarget(item);
      }
    }
  }
}

constexpr int64_t kMaxCalcDim = 1;
constexpr int64_t kCompareDim = std::numeric_limits<int64_t>::max();

static OptStrategy::CalcKind TensorComputable(PyObject *obj, ssize_t max_dim) {
  ShapeVector shape;
  if (tensor::IsTensorPy(obj)) {
    auto tensorPtr = tensor::ConvertToTensor(obj);
    shape = tensorPtr->shape();
  }

  if (!std::any_of(shape.begin(), shape.end(), [max_dim](const int64_t dim) { return dim > max_dim; })) {
    return OptStrategy::CalcKind::kCalcValue;
  }

  return OptStrategy::CalcKind::kCalcShape;
}

static OptStrategy::CalcKind ObjectComputable(PyObject *obj, ssize_t max_dim = kMaxCalcDim) {
  static const std::vector<bool (*)(PyObject *)> computable = {
    [](PyObject *op) { return op == Py_None || op == Py_True || op == Py_False || op == Py_Ellipsis; },
    CheckScalar,
    CheckContainer,
    [](PyObject *op) { return IsMsClass(reinterpret_cast<PyObject *>(Py_TYPE(op))); },
    IsNumpyObject,
  };
  if (obj == nullptr) {
    return OptStrategy::CalcKind::kCalcUnsupported;
  } else if (std::any_of(computable.begin(), computable.end(), [&obj](auto check) { return check(obj); })) {
    return OptStrategy::CalcKind::kCalcValue;
  } else if (IsTensorPyObject(obj)) {
    return TensorComputable(obj, max_dim);
  } else {
    return OptStrategy::CalcKind::kCalcUnsupported;
  }
}

using CheckPyObjectFunc = OptStrategy::CalcKind (*)(int bytecode, int opargs, const PyObjectArray &objs);

OptStrategy::CalcKind MakeCalcStrategyByObject(int bytecode, int opargs, const PyObjectArray &objs) {
  return ObjectComputable(objs[0]);
}

OptStrategy::CalcKind MakeInplaceCalcStrategyByObject(int bytecode, int opargs, const PyObjectArray &objs) {
  std::set<std::string> inplace = {"numpy.ndarray", "list", "<unnamed>"};
  const char *tp_name = Py_TYPE(objs[0])->tp_name ? (Py_TYPE(objs[0]))->tp_name : "<unnamed>";
  return inplace.find(tp_name) == inplace.end() ? ObjectComputable(objs[0]) : OptStrategy::CalcKind::kCalcUnsupported;
}

OptStrategy::CalcKind MakeCalcStrategyByMatMul(int bytecode, int opargs, const PyObjectArray &objs) {
  auto oc1 = ObjectComputable(objs[0]);
  auto oc2 = ObjectComputable(objs[1]);
  if (oc1 == OptStrategy::CalcKind::kCalcValue && oc2 == OptStrategy::CalcKind::kCalcValue) {
    return OptStrategy::CalcKind::kCalcValue;
  } else {
    return OptStrategy::CalcKind::kCalcUnsupported;
  }
}

OptStrategy::CalcKind MakeCalcStrategyByCompare(int bytecode, int opargs, const PyObjectArray &objs) {
  if (objs[0] == Py_None || objs[1] == Py_None) {
    return OptStrategy::CalcKind::kCalcValue;
  }
  if (py::isinstance<mindspore::Type>(objs[0]) || PyType_Check(objs[0])) {
    return OptStrategy::CalcKind::kCalcValue;
  }
  if (py::isinstance<mindspore::Type>(objs[1]) || PyType_Check(objs[1])) {
    return OptStrategy::CalcKind::kCalcValue;
  }
  auto oc1 = ObjectComputable(objs[0], kCompareDim);
  auto oc2 = ObjectComputable(objs[1], kCompareDim);
  if (oc1 == OptStrategy::CalcKind::kCalcValue && oc2 == OptStrategy::CalcKind::kCalcValue) {
    return OptStrategy::CalcKind::kCalcValue;
  } else if (oc1 == OptStrategy::CalcKind::kCalcUnsupported || oc2 == OptStrategy::CalcKind::kCalcUnsupported) {
    return OptStrategy::CalcKind::kCalcUnsupported;
  } else {
    return OptStrategy::CalcKind::kCalcShape;
  }
}

OptStrategy::CalcKind MakeCalcStrategyByGetItem(int bytecode, int opargs, const PyObjectArray &objs) {
  return IsTensorPyObject(objs[0]) ? OptStrategy::CalcKind::kCalcUnsupported : OptStrategy::CalcKind::kCalcValue;
}

static std::map<int, CheckPyObjectFunc> kBytecodeStrategy = {
  {UNARY_POSITIVE, MakeCalcStrategyByObject},
  {UNARY_NEGATIVE, MakeCalcStrategyByObject},
  {UNARY_NOT, MakeCalcStrategyByObject},
  {UNARY_INVERT, MakeCalcStrategyByObject},
  {BINARY_LSHIFT, MakeCalcStrategyByObject},
  {BINARY_RSHIFT, MakeCalcStrategyByObject},
  {BINARY_AND, MakeCalcStrategyByObject},
  {BINARY_XOR, MakeCalcStrategyByObject},
  {BINARY_OR, MakeCalcStrategyByObject},
  {BINARY_FLOOR_DIVIDE, MakeCalcStrategyByObject},
  {BINARY_TRUE_DIVIDE, MakeCalcStrategyByObject},
  {INPLACE_LSHIFT, MakeInplaceCalcStrategyByObject},
  {INPLACE_RSHIFT, MakeInplaceCalcStrategyByObject},
  {INPLACE_AND, MakeInplaceCalcStrategyByObject},
  {INPLACE_XOR, MakeInplaceCalcStrategyByObject},
  {INPLACE_OR, MakeInplaceCalcStrategyByObject},
  {INPLACE_FLOOR_DIVIDE, MakeInplaceCalcStrategyByObject},
  {INPLACE_TRUE_DIVIDE, MakeInplaceCalcStrategyByObject},
  {BINARY_POWER, MakeCalcStrategyByObject},
  {BINARY_ADD, MakeCalcStrategyByObject},
  {BINARY_SUBTRACT, MakeCalcStrategyByObject},
  {BINARY_MULTIPLY, MakeCalcStrategyByObject},
  {BINARY_MODULO, MakeCalcStrategyByObject},
  {INPLACE_POWER, MakeInplaceCalcStrategyByObject},
  {INPLACE_ADD, MakeInplaceCalcStrategyByObject},
  {INPLACE_SUBTRACT, MakeInplaceCalcStrategyByObject},
  {INPLACE_MULTIPLY, MakeInplaceCalcStrategyByObject},
  {INPLACE_MODULO, MakeInplaceCalcStrategyByObject},
  {BINARY_MATRIX_MULTIPLY, MakeCalcStrategyByMatMul},
  {INPLACE_MATRIX_MULTIPLY, MakeInplaceCalcStrategyByObject},
  {BINARY_SUBSCR, MakeCalcStrategyByGetItem},
  {COMPARE_OP, MakeCalcStrategyByCompare},
};

OptStrategy::CalcKind OptStrategy::MakeCalcStrategyByInputs(int bytecode, int opargs, const PyObjectArray &objs) {
  auto iter = kBytecodeStrategy.find(bytecode);
  if (iter != kBytecodeStrategy.end()) {
    return iter->second(bytecode, opargs, objs);
  }
  return CalcKind::kCalcUnsupported;
}

OptStrategy::CalcKind OptStrategy::MakeCalcStrategyByShape(const ShapeVector &shape) {
  if (!std::any_of(shape.begin(), shape.end(), [](const int64_t dim) { return dim > kMaxCalcDim; })) {
    return CalcKind::kCalcValue;
  } else {
    return CalcKind::kCalcShape;
  }
}

OptCodeSet OptStrategy::MakeGuardListStrategyByFrame(const OptCodeSet &codes) {
  OptCodeSet ret;
  std::transform(codes.begin(), codes.end(), std::back_inserter(ret), [](const OptCodePtr &code) { return code; });
  return ret;
}

GuardItemVector OptStrategy::MakeGuardItemListStrategyByFrame(const GuardItemVector &list) {
  GuardItemVector ret;
  std::transform(list.begin(), list.end(), std::back_inserter(ret), [](const GuardItemPtr &code) { return code; });
  return ret;
}
}  // namespace pijit
}  // namespace mindspore
