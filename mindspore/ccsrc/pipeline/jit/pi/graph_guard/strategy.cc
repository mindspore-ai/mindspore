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
#include "pipeline/jit/pi/graph_guard/strategy.h"
#include <algorithm>
#include <limits>
#include <string>
#include <vector>
#include <map>
#include "pybind11/pybind11.h"
#include "pybind_api/ir/primitive_py.h"
#include "include/common/utils/convert_utils_py.h"
#include "pipeline/jit/ps/pipeline.h"
#include "pipeline/jit/pi/utils/utils.h"
#include "pipeline/jit/pi/pydef.h"

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
  if (co != nullptr && static_cast<int>(PyBytes_GET_SIZE(co->co_code) / sizeof(_Py_CODEUNIT)) < threshold) {
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

void ShrinkCodeSet(OptCodeSet *set, OptCodePtr target) {
  OptCodeSet match, mismatch;
  auto guard_target = target->GetGuard();
  for (size_t i = set->size(); i != 0;) {
    i--;
    auto item = set->at(i);
    auto guard_item = item->GetGuard();
    if (guard_target->MatchShape(guard_item)) {
      match.insert(match.begin(), item);
    } else {
      mismatch.insert(mismatch.begin(), item);
    }
  }
  set->clear();
  set->insert(set->begin(), mismatch.begin(), mismatch.end());
  set->insert(set->end(), match.begin(), match.end());
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
        if (enable_dynamicshape) {
          ShrinkCodeSet(&set, except);
        }
        toDel.insert(toDel.begin(), set.begin() + limit_count, set.end());
        for (auto item : toDel) {
          hub->DelOptTarget(item);
        }
      }
    }
    if (limit_size > 0) {
      auto graph_executor = mindspore::pipeline::GraphExecutorPy::GetInstance();
      OptCodeSet toDel;
      if (enable_dynamicshape) {
        ShrinkCodeSet(&set, except);
      }
      for (auto item : set) {
        if (limit_size == 0) {
          toDel.push_back(item);
        }
        std::string phase = item->GetPhase();
        if (phase.size() > 0) {
          FuncGraphPtr ms_func_graph = graph_executor->GetFuncGraph(phase);
          int node_count = static_cast<int>(ms_func_graph->nodes().size());
          for (auto fg : ms_func_graph->func_graphs_used_total()) {
            node_count += static_cast<int>(fg->nodes().size());
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
  if (py::isinstance<mindspore::tensor::Tensor>(obj) || py::isinstance<mindspore::tensor::MetaTensor>(obj)) {
    auto tensor_ptr = py::cast<mindspore::tensor::MetaTensorPtr>(obj);
    auto shape = tensor_ptr->shape();
    if (!std::any_of(shape.begin(), shape.end(), [max_dim](const int64_t dim) { return dim > max_dim; })) {
      return OptStrategy::CalcKind::kCalcValue;
    }
  }
  return OptStrategy::CalcKind::kCalcShape;
}

static OptStrategy::CalcKind StubTensorComputable(PyObject *obj, ssize_t max_dim) {
  auto stub = PyObject_GetAttrString(obj, "stub");
  if (stub != nullptr && stub != Py_None) {
    auto pyObj = py::cast<py::object>(stub);
    auto ptr = pyObj.cast<mindspore::stub::StubNodePtr>();
    auto base = ptr->ToAbstract();
    auto shape = base->BuildShape()->cast<abstract::ShapePtr>();
    Py_DECREF(stub);
    if (shape && !shape->IsDynamic()) {
      if (!std::any_of(shape->shape().begin(), shape->shape().end(),
                       [max_dim](const int64_t dim) { return dim > max_dim; })) {
        return OptStrategy::CalcKind::kCalcValue;
      }
    } else {
      return OptStrategy::CalcKind::kCalcUnsupported;
    }
  } else {
    obj = PyObject_GetAttrString(obj, "tensor");
    auto pyObj = py::cast<py::object>(obj);
    Py_DECREF(obj);
    auto tensor_ptr = pyObj.cast<mindspore::tensor::TensorPtr>();
    auto shape = tensor_ptr->shape();
    if (!std::any_of(shape.begin(), shape.end(), [max_dim](const int64_t dim) { return dim > max_dim; })) {
      return OptStrategy::CalcKind::kCalcValue;
    }
  }
  return OptStrategy::CalcKind::kCalcShape;
}

static OptStrategy::CalcKind ObjectComputable(PyObject *obj, ssize_t max_dim = kMaxCalcDim) {
  if (obj == nullptr) {
    return OptStrategy::CalcKind::kCalcUnsupported;
  } else if (obj == Py_None || obj == Py_True || obj == Py_False || obj == Py_Ellipsis || CheckScalar(obj) ||
             CheckContainer(obj)) {
    return OptStrategy::CalcKind::kCalcValue;
  } else if (IsTensorPyObject(obj)) {
    return TensorComputable(obj, max_dim);
  } else if (IsStubTensor(py::cast<py::object>(obj))) {
    return StubTensorComputable(obj, max_dim);
  } else {
    return OptStrategy::CalcKind::kCalcUnsupported;
  }
}

using CheckPyObjectFunc = OptStrategy::CalcKind (*)(int bytecode, int opargs, const PyObjectArray &objs);

OptStrategy::CalcKind MakeCalcStrategyByObject(int bytecode, int opargs, const PyObjectArray &objs) {
  return ObjectComputable(objs[0]);
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
  {INPLACE_LSHIFT, MakeCalcStrategyByObject},
  {INPLACE_RSHIFT, MakeCalcStrategyByObject},
  {INPLACE_AND, MakeCalcStrategyByObject},
  {INPLACE_XOR, MakeCalcStrategyByObject},
  {INPLACE_OR, MakeCalcStrategyByObject},
  {INPLACE_FLOOR_DIVIDE, MakeCalcStrategyByObject},
  {INPLACE_TRUE_DIVIDE, MakeCalcStrategyByObject},
  {BINARY_POWER, MakeCalcStrategyByObject},
  {BINARY_ADD, MakeCalcStrategyByObject},
  {BINARY_SUBTRACT, MakeCalcStrategyByObject},
  {BINARY_MULTIPLY, MakeCalcStrategyByObject},
  {BINARY_MODULO, MakeCalcStrategyByObject},
  {INPLACE_POWER, MakeCalcStrategyByObject},
  {INPLACE_ADD, MakeCalcStrategyByObject},
  {INPLACE_SUBTRACT, MakeCalcStrategyByObject},
  {INPLACE_MULTIPLY, MakeCalcStrategyByObject},
  {INPLACE_MODULO, MakeCalcStrategyByObject},
  {BINARY_MATRIX_MULTIPLY, MakeCalcStrategyByMatMul},
  {INPLACE_MATRIX_MULTIPLY, MakeCalcStrategyByMatMul},
  {BINARY_SUBSCR,
   [](int bytecode, int opargs, const PyObjectArray &objs) { return OptStrategy::CalcKind::kCalcValue; }},
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

OptCodeSet OptStrategy::MakeGuardListStrategyByFrame(const PyFrameObject *frame, const OptCodeSet &codes) {
  OptCodeSet ret;
  for (auto code : codes) {
    ret.push_back(code);
  }
  return ret;
}

GuardItemVector OptStrategy::MakeGuardItemListStrategyByFrame(const PyFrameObject *frame, const GuardItemVector &list) {
  GuardItemVector ret;
  for (auto item : list) {
    ret.push_back(item);
  }
  return ret;
}
}  // namespace pijit
}  // namespace mindspore
