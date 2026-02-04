/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_UTILS_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_UTILS_H_

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <tuple>
#include <Python.h>
#include "pynative/utils/base.h"
#include "include/pynative/utils/pynative_execute.h"
#include "include/pynative/utils/pyboost/op_runner.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/forward/pyboost/forward_task.h"
#include "include/utils/tensor_py.h"
#include "include/utils/visible.h"

namespace mindspore {
namespace pynative {
class PyNativeExecutor;
namespace PyNativeAlgo {
// Common function
struct Common {
  static std::string GetIdByValue(const ValuePtr &v);
  static void SplitString(const std::string &str, std::vector<std::string> *id_vec);
  static bool IsTensor(const ValuePtr &v, bool include_sequence = false);
  static bool IsControlFlowGraph(const FuncGraphPtr &func_graph);
  static ValuePtr FilterSensValues(const ValuePtr &value, bool dict_convert_to_tuple);
  static tensor::TensorPtr GetTensorFromParam(const AnfNodePtr &param_node);
  static const std::shared_ptr<PyNativeExecutor> &GetPyNativeExecutor();
  static ValuePtr StubNodeToValue(const ValuePtr &val);
  static void StubNodeToValue(const FrontendOpRunInfoPtr &op_run_info);
  static tensor::TensorPtr StubNodeToTensor(const ValuePtr &value);
  PYNATIVE_EXPORT static tensor::TensorPtr ConvertStubNodeToTensor(const ValuePtr &v, bool need_contiguous,
                                                                   bool requires_grad, bool is_inplace = false);
  PYNATIVE_EXPORT static std::optional<tensor::TensorPtr> ConvertStubNodeToTensor(const std::optional<ValuePtr> &v,
                                                                                  bool need_contiguous,
                                                                                  bool requires_grad,
                                                                                  bool is_inplace = false);
  static ValueTuplePtr ConvertStubNodeToValueTuple(const ValueListPtr &v, bool need_contiguous, bool requires_grad,
                                                   bool is_inplace = false);
  static ValueTuplePtr ConvertStubNodeToValueTuple(const ValueTuplePtr &v, bool need_contiguous, bool requires_grad,
                                                   bool is_inplace = false);
  static std::optional<ValueTuplePtr> ConvertStubNodeToValueTuple(const std::optional<ValueTuplePtr> &v,
                                                                  bool need_contiguous, bool requires_grad,
                                                                  bool is_inplace = false);
  static ValueNodePtr CreateValueNodeByValue(const ValuePtr &v, const abstract::AbstractBasePtr &abs = nullptr);
  static void SetOutputUsedInBpropGraph(const ValuePtr &value);
  static ValuePtr CreateFakeValueWithoutDeviceAddress(const ValuePtr &value);
  static void ClearDeviceAddress(const ValuePtr &value);
  static inline bool IsConstant(InputType grad_type) { return grad_type == InputType::kConstant; }
  static void SetGraphInputAndWeightsInfo(const FrontendOpRunInfoPtr &op_run_info, const FuncGraphPtr &func_graph);
  static tensor::TensorPtr CaculateGradNorm(const tensor::TensorPtr &grad);
  template <typename T>
  static std::string PrintDebugInfo(const std::vector<T> &items, const std::string &info_header = "",
                                    bool is_print_tensor_data = false) {
    static constexpr size_t end_char_size = 2;
    std::ostringstream buf;
    buf << info_header;
    for (size_t i = 0; i < items.size(); ++i) {
      if (items[i] == nullptr) {
        MS_LOG(DEBUG) << "The " << i << "'th item is nullptr!";
        continue;
      }
      if (items[i]->template isa<tensor::Tensor>() && is_print_tensor_data) {
        auto tensor = items[i]->template cast<tensor::TensorPtr>();
        auto grad = std::make_shared<tensor::Tensor>(*tensor);
        auto norm_val = CaculateGradNorm(grad);
        auto norm_val_cpu = norm_val->cpu();
        buf << i << "th: "
            << "ptr " << items[i].get() << ", " << norm_val_cpu->ToStringRepr() << ", ";
      } else {
        buf << i << "th: "
            << "ptr " << items[i].get() << ", " << items[i]->ToString() << ", ";
      }
    }
    return buf.str().erase(buf.str().size() - end_char_size);
  }
  static bool IsHookNeedSaveInputs(const PrimitivePyPtr &prim);
  static bool IsVmOp(const std::string &op_name);
  static std::vector<int64_t> BuildShape(const abstract::AbstractBasePtr &abs);
  static void ClearRes();
  static void DoGradInner(runtime::OpRunnerInfo *op_runner_info, VectorRef *op_outputs);
  static tensor::TensorPtr GetTensorFromSparseTensor(const ValuePtr &val);
  static void WaitBprop();
};

// Parser python
struct PyParser {
  static std::string GetIdByPyObj(const py::object &obj);
  static void SetPrim(const FrontendOpRunInfoPtr &op_run_info, const py::object &prim_arg);
  static void ParseOpInputByPythonObj(const FrontendOpRunInfoPtr &op_run_info, const py::list &op_inputs,
                                      bool stub = false);
  static std::string BuilidPyInputTypeString(const py::object &obj);
  static std::string BuildPyObjectInputTypeString(PyObject *obj);
  static void PrintTypeCastError(const ops::OpDefPtr &op_def, const py::list &op_inputs, size_t idx);
  static void PrintTypeCastErrorForPyObject(const ops::OpDefPtr &op_def, PyObject *op_inputs, size_t idx);
};

// Data convert
struct DataConvert {
  static void GetInputTensor(const FrontendOpRunInfoPtr &op_run_info);
  static void ConvertCSRTensorToTensorList(const FrontendOpRunInfoPtr &op_run_info,
                                           const tensor::CSRTensorPtr &csr_tensor, size_t index);
  static void ConvertMapTensor(const FrontendOpRunInfoPtr &op_run_info, const tensor::MapTensorPtr &map_tensor,
                               size_t index);
  static ValuePtr ConvertValueDictToValueTuple(const ValuePtr &v);
  static void PlantTensorTupleToVector(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq,
                                       size_t index);
  static void ConvertTupleValueToTensor(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq,
                                        size_t index);
  static void MarkInputs(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v, size_t index);
  static bool RunOpConvertConstInputToAttr(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v,
                                           size_t input_index);
  static ValuePtr ValueListToValue(const ValuePtrList &values, const abstract::AbstractBasePtr &abs);
  static ValuePtrList TensorListToValueList(const tensor::TensorPtrList &tensor_list);
};

struct PyBoost {
  static PyboostOpRunInfoPtr Init_Pyboost(const PrimitivePtr &prim);
  static FrontendOpRunInfoPtr Init(const PrimitivePtr &prim);
  static void DoGrad(const kernel::pyboost::OpPtr &op, const OpGradInfoPtr &grad_info, const AsyncStatus &async_status);
  static void DoGrad(const OpGradInfoPtr &grad_info, const AsyncStatus &async_status);
  PYNATIVE_EXPORT static void MarkSideEffect(PyObject *arg);
  static void UpdateStubOutput(const kernel::pyboost::OpPtr &op, const stub::StubNodePtr &stub_output,
                               const AbstractBasePtr &abstract, const ValuePtr &real_out);
  static PrimitivePtr ConvertPrimitive(const py::object &obj);
  static py::object RunPyFunction(const PrimitivePtr &prim, const py::list &args);
  template <typename T>
  static ValuePtr OptionalToValue(const std::optional<T> &val) {
    if (!val.has_value()) {
      return kNone;
    }
    return val.value();
  }

  template <typename Tuple, size_t... N>
  static std::vector<ValuePtr> TupleToVector(const Tuple &tuple, std::index_sequence<N...>) {
    std::vector<ValuePtr> inputs;
    ((void)inputs.emplace_back(OptionalToValue(std::get<N>(tuple))), ...);
    return inputs;
  }

  template <typename T>
  static T OptionalToValue(const T &val) {
    return val;
  }

  template <size_t N, typename... T>
  static auto SetPyBoostCastForInputs(const PyboostOpRunInfoPtr &op_run_info, const std::string &op_name,
                                      const std::vector<std::vector<size_t>> &same_type_table, T... t) {
    MS_EXCEPTION_IF_NULL(op_run_info);
    const size_t &input_size = sizeof...(t);
    if (op_name == kCast) {
      return std::make_tuple(t...);
    }
    const auto &pyboost_cast_operation = Common::GetPyNativeExecutor()->forward_executor()->pyboost_cast_operation();
    const auto &ret = pyboost_cast_operation->DoMixPrecisionCast(op_run_info, input_size, t...);
    if constexpr (N != 0) {
      return pyboost_cast_operation->DoImplicitCast<N>(op_run_info, input_size, same_type_table, ret);
    }
    return ret;
  }

  static void MarkPyBoostInputs(const OpGradInfoPtr &op_grad_info);
  static void BumpVersionAsync(tensor::Version version);
  static void UpdateVersionAsync(const autograd::ViewAutoGradMetaDataPtr &grad_meta, const tensor::Version &version);
  static ValuePtr OutputToValue(const tensor::TensorPtr &output) { return output; }
  static ValuePtr MultiOutputToValue(const std::vector<TensorPtr> &outputs) {
    std::vector<ValuePtr> output_values;
    output_values.reserve(outputs.size());
    (void)std::transform(outputs.begin(), outputs.end(), std::back_inserter(output_values),
                         [](const TensorPtr &value) -> ValuePtr { return value; });
    return std::make_shared<ValueTuple>(output_values);
  }

  template <typename... Args>
  static ValuePtr MultiOutputToValue(const std::tuple<Args...> &outputs) {
    std::vector<ValuePtr> output_values = TupleToVector(outputs);
    return std::make_shared<ValueTuple>(output_values);
  }

 private:
  template <std::size_t... Is, typename Tuple>
  static std::vector<ValuePtr> UnpackTuple(const Tuple &t, std::index_sequence<Is...>) {
    return {std::get<Is>(t)...};
  }

  template <typename... Args>
  static std::vector<ValuePtr> TupleToVector(const std::tuple<Args...> &t) {
    return UnpackTuple(t, std::index_sequence_for<Args...>{});
  }
};
};  // namespace PyNativeAlgo

class PYNATIVE_EXPORT OpCall {
 public:
  OpCall() = default;
  OpCall(std::string name, std::function<py::object(const py::args &, const py::kwargs &)> func)
      : name_(std::move(name)), func_(std::move(func)) {}

  py::object operator()(const py::args &args, const py::kwargs &kwargs) {
    MS_LOG(DEBUG) << "In OpCall __call__";
    if (func_) {
      return func_(args, kwargs);
    } else {
      MS_LOG(EXCEPTION) << "OpCall func is not callable!";
    }
  }

  const std::string &name() const { return name_; }

 private:
  std::string name_;
  std::function<py::object(const py::args &, const py::kwargs &)> func_;
};

class PYNATIVE_EXPORT TensorOverloadCall {
 public:
  TensorOverloadCall() = default;
  TensorOverloadCall(std::string name,
                     std::function<py::object(const py::object &, const py::args &, const py::kwargs &)> func)
      : name_(std::move(name)), func_(std::move(func)) {}

  py::object operator()(const py::object &self, const py::args &args, const py::kwargs &kwargs) {
    MS_LOG(DEBUG) << "In OpCall __call__";
    if (func_) {
      return func_(self, args, kwargs);
    } else {
      MS_LOG(EXCEPTION) << "OpCall func is not callable!";
    }
  }

  const std::string &name() const { return name_; }

 private:
  std::string name_;
  std::function<py::object(const py::object &, const py::args &, const py::kwargs &)> func_;
};

PYNATIVE_EXPORT void RegOpCall(const py::module *m);

PYNATIVE_EXPORT void DispatchOp(const std::shared_ptr<runtime::AsyncTask> &task);
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_UTILS_H_
