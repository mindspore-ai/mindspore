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
#include "pipeline/pynative/base.h"
#include "pipeline/pynative/pynative_execute.h"
#include "kernel/pyboost/op_runner.h"
#include "kernel/pyboost/op_register.h"
#include "pipeline/pynative/forward/forward_task.h"
#include "pipeline/pynative/grad/function/func_builder.h"
#include "pipeline/jit/ps/parse/data_converter.h"

namespace mindspore {
namespace pynative {
class PyNativeExecutor;
using CallBackFn = std::function<VectorRef(const VectorRef &arg_list)>;
enum class SpecialType { kZerosLikeType = 0, kOnesLikeType = 1 };

namespace PyNativeAlgo {
// Common function
struct Common {
  static AbstractBasePtr SetAbstractValueToAnyValue(const AbstractBasePtr &abs);
  static AnfNodePtr ConvertValueSequenceToMakeTuple(const ValueNodePtr &node, const FuncGraphPtr &func_graph);
  static std::string GetIdByValue(const ValuePtr &v);
  static std::string GetCellId(const std::string &obj_id, const std::vector<std::string> &input_arg_id_vec,
                               const std::vector<ValuePtr> &input_arg_value_vec);
  static void SplitString(const std::string &str, std::vector<std::string> *id_vec);
  static bool ValueHasDynamicShape(const ValuePtr &value);
  static bool IsTensor(const ValuePtr &v, bool include_sequence = false);
  static bool IsControlFlowGraph(const FuncGraphPtr &func_graph);
  static ValuePtr FilterSensValues(const ValuePtr &value, bool dict_convert_to_tuple);
  static tensor::BaseTensorPtr GetTensorFromParam(const AnfNodePtr &param_node);
  static void DumpGraphIR(const std::string &filename, const FuncGraphPtr &graph);
  static TypeId GetTypeFromAbstract(const abstract::AbstractBasePtr &abs);
  static ShapeVector GetShapeFromAbstract(const abstract::AbstractBasePtr &abs);
  static std::pair<TypePtr, TypeId> GetTypeFromValue(const ValuePtr &v);
  static ShapeVector GetShapeFromValue(const ValuePtr &v);
  static ValuePtr CreatOutputTensorValueByAbstract(const abstract::AbstractBasePtr &abs);
  static void ReplaceCNodeWithValueNode(const FuncGraphPtr &bprop_graph);
  static const std::shared_ptr<PyNativeExecutor> &GetPyNativeExecutor();
  static void StubNodeToValue(const FrontendOpRunInfoPtr &op_run_info);
  static tensor::BaseTensorPtr StubNodeToTensor(const ValuePtr &value);
  static tensor::BaseTensorPtr ConvertStubNodeToTensor(const ValuePtr &v, bool need_contiguous, bool requires_grad);
  static std::optional<tensor::BaseTensorPtr> ConvertStubNodeToTensor(const std::optional<ValuePtr> &v,
                                                                      bool need_contiguous, bool requires_grad);
  static ValueTuplePtr ConvertStubNodeToValueTuple(const ValueListPtr &v, bool need_contiguous, bool requires_grad);
  static ValueTuplePtr ConvertStubNodeToValueTuple(const ValueTuplePtr &v, bool need_contiguous, bool requires_grad);
  static std::optional<ValueTuplePtr> ConvertStubNodeToValueTuple(const std::optional<ValueTuplePtr> &v,
                                                                  bool need_contiguous, bool requires_grad);
  static void GetConstInputToAttr(const PrimitivePtr &op_prim, const std::string &op_name,
                                  const std::string &device_target, bool is_dynamic_shape,
                                  mindspore::HashSet<size_t> *input_to_attr_index);
  static ValueNodePtr CreateValueNodeByValue(const ValuePtr &v, const abstract::AbstractBasePtr &abs = nullptr);
  static void SetOutputUsedInBpropGraph(const ValuePtr &value);
  static ValuePtr CreateFakeValueWithoutDeviceAddress(const ValuePtr &value);
  static tensor::TensorPtr CreateFakeTensorWithoutDeviceAddress(const tensor::TensorPtr &tensor);
  static inline bool IsParam(InputType grad_type) {
    return grad_type == InputType::kParameter || grad_type == InputType::kInput;
  }
  static inline bool IsParamRequiresGrad(const tensor::BaseTensorPtr &tensor) {
    return tensor->param_info() != nullptr && tensor->param_info()->requires_grad();
  }
  static void ClearDeviceAddress(const ValuePtr &value);
  static inline bool IsConstant(InputType grad_type) { return grad_type == InputType::kConstant; }
  static InputType SetValueGradInfo(const ValuePtr &value, const TopCellInfoPtr &top_cell, InputType grad_type);
  static InputType SetTensorGradInfo(const tensor::BaseTensorPtr &tensor, const TopCellInfoPtr &top_cell);
  static void SetGraphInputAndWeightsInfo(const FrontendOpRunInfoPtr &op_run_info, const FuncGraphPtr &func_graph,
                                          const TopCellInfoPtr &top_cell);
  static void ProcessTupleParam(const FuncGraphPtr &bprop_graph, size_t position);
  static void ProcessDictParam(const FuncGraphPtr &bprop_graph, size_t position);
  static void FreeFuncGraphForwardNodes(const FuncGraphPtr &func_graph);
  static tensor::BaseTensorPtr ConvertToContiguousTensor(const tensor::BaseTensorPtr &tensor, bool requires_grad);
  static ValuePtr ConvertToContiguousValue(const ValuePtr &v, bool requires_grad);
  static size_t GetValueSize(const ValuePtr &v);
  static ValuePtr CreateTensorByConstantValue(const ValuePtr &value);

  template <typename T>
  static std::string PrintDebugInfo(std::vector<T> items, const std::string &info_header = "",
                                    bool is_print_tensor_data = false) {
    static constexpr size_t end_char_size = 2;
    std::ostringstream buf;
    buf << info_header;
    for (size_t i = 0; i < items.size(); ++i) {
      if (items[i] == nullptr) {
        MS_LOG(DEBUG) << "The " << i << "'th item is nullptr!";
        continue;
      }
      if (items[i]->template isa<tensor::BaseTensor>() && is_print_tensor_data) {
        auto tensor = items[i]->template cast<tensor::BaseTensorPtr>();
        auto grad = std::make_shared<tensor::Tensor>(*tensor);
        grad->data_sync();
        buf << i << "th: "
            << "ptr " << items[i].get() << ", " << grad->ToStringRepr() << ", ";
      } else {
        buf << i << "th: "
            << "ptr " << items[i].get() << ", " << items[i]->ToString() << ", ";
      }
    }
    return buf.str().erase(buf.str().size() - end_char_size);
  }
};

// Parser python
struct PyParser {
  static std::string GetIdByPyObj(const py::object &obj);
  static std::pair<std::vector<std::string>, std::vector<ValuePtr>> GetArgsIdAndValue(const py::args &args);
  static void SetPrim(const FrontendOpRunInfoPtr &op_run_info, const py::object &prim_arg);
  static void ParseOpInputByPythonObj(const FrontendOpRunInfoPtr &op_run_info, const py::list &op_inputs,
                                      bool stub = false);
  static std::string BuilidPyInputTypeString(const py::object &obj);

  static inline bool IsSupportTensorCast(const std::vector<ops::OP_DTYPE> &cast_types) {
    for (const auto &type : cast_types) {
      if (type == ops::DT_TENSOR) {
        return true;
      }
    }
    return false;
  }
  static void PrintTypeCastError(const ops::OpDefPtr &op_def, const py::list &op_inputs, size_t idx);
};

// Data convert
struct DataConvert {
  static py::object ValueToPyObj(const ValuePtr &v);
  static ValuePtr PyObjToValue(const py::object &obj, bool stub = false);
  static ValuePtr BaseRefToValue(const BaseRef &value, bool requires_grad, bool is_out_sequence);
  static ValuePtr VectorRefToValue(const VectorRef &vec_ref, bool requires_grad, bool is_out_sequence);
  static void FlattenValueSeqArg(const ValuePtr &v, bool is_only_flatten_tensor_seq, bool is_filter_tensor,
                                 std::vector<ValuePtr> *flatten_v);
  static void FlattenArgs(const std::vector<ValuePtr> &v_vec, std::vector<ValuePtr> *flatten_v, bool has_sens);
  static ValuePtrList FlattenTensorSeqInValue(const ValuePtr &v);
  static ValuePtrList FlattenTensorSeqInValueSeq(const ValuePtrList &v, bool only_flatten_tensor = true);
  static void GetInputTensor(const FrontendOpRunInfoPtr &op_run_info, const TopCellInfoPtr &top_cell);
  static void ConvertCSRTensorToTensorList(const FrontendOpRunInfoPtr &op_run_info,
                                           const tensor::CSRTensorPtr &csr_tensor, const TopCellInfoPtr &top_cell,
                                           size_t index);
  static void ConvertMapTensor(const FrontendOpRunInfoPtr &op_run_info, const tensor::MapTensorPtr &map_tensor,
                               const TopCellInfoPtr &top_cell, size_t index);
  static ValuePtr ConvertValueDictToValueTuple(const ValuePtr &v);
  static void PlantTensorTupleToVector(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq,
                                       size_t index, const TopCellInfoPtr &top_cell);
  static void ConvertValueTensorId(const ValuePtr &value, std::vector<std::string> *converted_tensor_id);
  static void ConvertTupleValueToTensor(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq,
                                        size_t index, const TopCellInfoPtr &top_cell);
  static void MarkInputs(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v, size_t index,
                         const TopCellInfoPtr &top_cell);
  static bool RunOpConvertConstInputToAttr(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v,
                                           size_t input_index);
};

struct PyBoost {
  static FrontendOpRunInfoPtr Init(const PrimitivePtr &prim, const py::list &args);
  static void MakeOutputValue(const FrontendOpRunInfoPtr &op_run_info, const kernel::pyboost::OpPtr &op);
  static void DoGrad(const kernel::pyboost::OpPtr &op, const FrontendOpRunInfoPtr &op_run_info,
                     ValuePtrList &&op_inputs);
  static void SetAnyValueForAbstract(const kernel::pyboost::OpPtr &op);
  static void UpdateStubOutput(const FrontendOpRunInfoPtr &op_run_info, const AbstractBasePtr &abstract,
                               const kernel::pyboost::OpPtr &op);
  static void UpdateOpRunInfo(const kernel::pyboost::OpPtr &op, const FrontendOpRunInfoPtr &op_run_info);
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
  static auto SetPyBoostCastForInputs(const FrontendOpRunInfoPtr &op_run_info,
                                      const std::vector<std::vector<size_t>> &same_type_table, T... t) {
    MS_EXCEPTION_IF_NULL(op_run_info);
    op_run_info->input_size = sizeof...(t);
    if (op_run_info->op_grad_info->op_prim->name() == kCast) {
      return std::make_tuple(t...);
    }
    const auto &pyboost_cast_operation = Common::GetPyNativeExecutor()->forward_executor()->pyboost_cast_operation();
    const auto &ret = pyboost_cast_operation->DoMixPrecisionCast(op_run_info, t...);
    if constexpr (N != 0) {
      return pyboost_cast_operation->DoImplicitCast<N>(op_run_info, same_type_table, ret);
    }
    return ret;
  }
  static void DataSyncForGraph(const kernel::pyboost::OpPtr &op, ValuePtrList &&op_inputs);
};

// Used for auto grad, like func_grad and ir grad
struct AutoGrad {
  static bool IsPrimNeedGrad(const PrimitivePtr &prim);
  static bool NeedGrad(const std::vector<ValuePtr> &input_values);
  static bool IsZerosLikeNode(const AnfNodePtr &node);
  static ValuePtr GetFakeZeroTensor();
  static ValuePtr BuildSpecialValueGrad(const ValuePtr &value, const tensor::BaseTensorPtr &grad,
                                        autograd::FuncBuilder *func_builder, const SpecialType &type);
  static AnfNodePtr BuildSpecialNode(const KernelGraphPtr &tape, const ValuePtr &value,
                                     const abstract::AbstractBasePtr &abs, const SpecialType &type);
  static AnfNodePtr BuildSparseTensorNode(const KernelGraphPtr &tape, const ValuePtr &sparse_value,
                                          const AnfNodePtr &dout_value_node);
  static void SetGradMetaData(const ValuePtr &value, const VariablePtr &variable, const ParameterPtr &param = nullptr);
  static void SetGradInfoForInputs(const ValuePtr &value, const VariablePtr &variable,
                                   const ParameterPtr &param = nullptr);

  // Create fake bprop
  static void BuildFakeBpropCNode(const CNodePtr &cnode, std::vector<CNodePtr> *outputs);
  static CallBackFn CreateGraphCallBack(const FuncGraphPtr &call_graph, const std::string &cache_key,
                                        const GraphCallCondition &graph_call_condition);
  static PrimitivePyPtr BuildBpropCutPrim(const PrimitivePtr &prim, bool is_need_recompute = false);
  static void CheckRecomputeInputs(const GradParamPtr &grad_param);
  static TopCellInfoPtr FindPreTopcell(const GradExecutor *grad_executor, const OpGradInfoPtr &op_grad_info,
                                       const std::string &op_info, const ValuePtr &value);
  static void UpdateGradOpInfo(const GradExecutor *grad_executor, const OpGradInfoPtr &op_grad_info,
                               const TopCellInfoPtr &pre_top_cell, bool is_jit_graph);
  static void ClearAutoGradStaticCache();
  static void CheckAndSetAbstract(const OpGradInfoPtr &op_grad_info);
  static void CacheOutputAbstract(const ValuePtr &v, const abstract::AbstractBasePtr &abs);
};

// Some common functions used in both jit and PackFunc grad
struct GradCommon {
  static bool IsRealOp(const AnfNodePtr &cnode);
  static void GetUsedCNodeInBpropGraph(const CNodePtr &cnode, const mindspore::HashSet<size_t> &unused_inputs,
                                       AnfNodePtrList *node_list);
  static void SetForward(const AnfNodePtrList &node_list);
};
};  // namespace PyNativeAlgo

void DispatchOp(const std::shared_ptr<runtime::AsyncTask> &task);
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_UTILS_H_
