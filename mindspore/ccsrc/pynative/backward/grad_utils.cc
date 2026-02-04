/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_GRAD_UTILS_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_GRAD_UTILS_H_

#include "pynative/backward/grad_utils.h"
#include <string>
#include <memory>
#include <utility>
#include <algorithm>
#include <vector>

#include "ir/tensor_new.h"
#include "primitive/sparse_ops.h"
#include "primitive/sequence_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/array_op_name.h"
#include "include/frontend/operator/primitive_py.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "pynative/backward/hook/hook_py.h"
#include "utils/ms_context.h"
#include "include/utils/utils.h"
#include "frontend/jit/ps/pipeline.h"
#include "include/utils/convert_utils_py.h"
#include "include/utils/pynative/common_utils.h"
#include "include/frontend/jit/ps/pass_interface.h"
#include "include/frontend/jit/ps/action_interface.h"
#include "include/frontend/jit/ps/resource_interface.h"
#include "include/frontend/optimizer/environ_conversion.h"
#include "include/frontend/optimizer/fallback_rewriter_opt.h"
#include "pynative/backward/jit_grad/jit_grad.h"
#include "primitive/sequence_op_name.h"
#include "primitive/structure_ops.h"
#include "primitive/other_ops.h"
#include "include/utils/pynative/abstract_converter.h"
#include "include/pynative/utils/pyboost/auto_generate/clone.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/pynative/utils/pynative_utils.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_o.h"
#include "primitive/auto_generate/gen_ops_primitive_z.h"
#include "pynative/backward/op_grad/func_grad.h"
#include "mindspore/ccsrc/include/frontend/optimizer/optimizer.h"
#include "include/frontend/optimizer/ad/grad_interface.h"
#include "include/pynative/utils/pyboost/grad_functions/pyboost_grad_functions.h"
#include "utils/device_manager_conf.h"
#include "ir/func_graph_flag.h"
#include "pynative/backward/op_grad/auto_generate/pyboost_native_grad_functions.h"
#include "include/runtime/utils/dispatch/dispatch_env.h"

namespace mindspore {
namespace pynative {
constexpr char kGrad[] = "grad";
using CallBackFn = std::function<VectorRef(const VectorRef &arg_list)>;
using VmEvalPtr = std::shared_ptr<std::function<BaseRef(const VectorRef &)>>;
const mindspore::HashSet<std::string> kGradBlackList{kMakeTupleOpName,         kMakeListOpName,
                                                     kTupleGetItemOpName,      kStopGradientOpName,
                                                     kUpdateStateOpName,       kNPUAllocFloatStatusOpName,
                                                     kNPUGetFloatStatusOpName, kNPUClearFloatStatusOpName};
mindspore::HashMap<std::string, pipeline::ResourcePtr> jit_call_graph_compile_cache_;

// for simply infer (simple infer will push abs in bprop queue)
static AbstractConverter kGradAbstractConverter;
using AutoGradMetaData = autograd::AutoGradMetaData;
using ViewAutoGradMetaData = autograd::ViewAutoGradMetaData;
using ViewAutoGradMetaDataPtr = std::shared_ptr<ViewAutoGradMetaData>;
using ViewInfo = autograd::ViewInfo;

class FuncRegister {
 public:
  FuncRegister() {
    kernel::pyboost::RegisterCloneFunc(AutoGradUtil::CheckAndCloneInplaceInput);
    runtime::RegisterDoGradFunc(PyNativeAlgo::Common::DoGradInner);
    RegisterWaitBpropFunc(PyNativeAlgo::Common::WaitBprop);
  }
};
static FuncRegister func_register;

namespace {
ValuePtr WrapCOOTensor(const ValuePtr &coo_out, const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(coo_out);
  auto coo_tensor = coo_out->cast<tensor::COOTensorPtr>();
  MS_EXCEPTION_IF_NULL(coo_tensor);
  auto value_tensor = value->cast<tensor::TensorPtr>();
  if (value_tensor == nullptr) {
    auto base_tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(base_tensor);
    value_tensor = std::make_shared<tensor::Tensor>(*base_tensor);
  }
  auto indices_tensor = coo_tensor->GetIndices();
  auto shape_vector = coo_tensor->shape();
  return std::make_shared<tensor::COOTensor>(indices_tensor, value_tensor, shape_vector);
}

ValuePtr WrapCSRTensor(const ValuePtr &csr_out, const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(csr_out);
  auto csr_tensor = csr_out->cast<tensor::CSRTensorPtr>();
  MS_EXCEPTION_IF_NULL(csr_tensor);
  auto value_tensor = value->cast<tensor::TensorPtr>();
  if (value_tensor == nullptr) {
    auto base_tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(base_tensor);
    value_tensor = std::make_shared<tensor::Tensor>(*base_tensor);
  }
  auto indptr_tensor = csr_tensor->GetIndptr();
  auto indices_tensor = csr_tensor->GetIndices();
  auto shape_vector = csr_tensor->shape();
  return std::make_shared<tensor::CSRTensor>(indptr_tensor, indices_tensor, value_tensor, shape_vector);
}

void ConvertSimpleInferInfoToAbstract(const OpGradInfoPtr &op_grad_info) {
  MS_EXCEPTION_IF_NULL(op_grad_info);
  // Get inputs abstract
  for (const auto &v : op_grad_info->input_value) {
    op_grad_info->input_abs.emplace_back(kGradAbstractConverter.ConvertAbstract(v));
  }

  // Get output abstract
  MS_EXCEPTION_IF_NULL(op_grad_info->output_value_simple_info);
  op_grad_info->out_abs = TransformValueSimpleInfoToAbstract(*op_grad_info->output_value_simple_info);

  // Set abstract to tensor
  AutoGradUtil::CacheOutputAbstract(op_grad_info->out_value, op_grad_info->out_abs);
  MS_LOG(DEBUG) << "Get output abstract " << op_grad_info->out_abs->ToString();
}

InputType SetValueGradInfoForTensor(const ValuePtr &value, InputType grad_type) {
  const auto &tensor_value = value->cast<tensor::TensorPtr>();
  auto auto_grad_meta_data = autograd::impl::GetAutogradMetaImpl(tensor_value);
  if (auto_grad_meta_data != nullptr) {
    if (auto_grad_meta_data->input_type() == InputType::kOpOutput) {
      return auto_grad_meta_data->input_type();
    }
    MS_LOG(DEBUG) << "Set input type for tensor " << tensor_value->id();
  } else if (grad_type != InputType::kConstant || tensor_value->is_parameter()) {
    MS_LOG(DEBUG) << "Create new auto grad meta for tensor " << tensor_value->id();
    auto_grad_meta_data = std::make_shared<AutoGradMetaData>();
    tensor_value->set_auto_grad_meta_data(auto_grad_meta_data);
  }
  // Scalar tensor auto grad meta data is nullptr
  if (auto_grad_meta_data == nullptr) {
    return grad_type;
  }
  if (tensor_value->is_parameter() && grad_type != InputType::kInput) {
    grad_type = InputType::kParameter;
    if (AutoGradUtil::IsParamRequiresGrad(tensor_value) && auto_grad_meta_data->UnsafeGetGradNodeImpl() == nullptr) {
      auto fn = std::make_shared<autograd::LeafNode>(tensor_value->param_info()->name(), tensor_value,
                                                     tensor_value->shape(), tensor_value->Dtype());
      auto_grad_meta_data->set_requires_grad(true);
      auto_grad_meta_data->set_grad_node(fn);
    }
  }
  auto_grad_meta_data->set_input_type(grad_type);
  if (grad_type == InputType::kInput && auto_grad_meta_data->UnsafeGetGradNodeImpl() == nullptr) {
    MS_LOG(DEBUG) << "Build leaf node for input";
    auto fn = std::make_shared<autograd::LeafNode>("input_" + std::to_string(tensor_value->id()), tensor_value,
                                                   tensor_value->shape(), tensor_value->Dtype(), false);
    auto_grad_meta_data->set_requires_grad(true);
    auto_grad_meta_data->set_grad_node(fn);
  }
  return grad_type;
}
}  // namespace

InputType AutoGradUtil::SetValueGradInfo(const ValuePtr &value, InputType grad_type) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>()) {
    return SetValueGradInfoForTensor(value, grad_type);
  }
  if (value->isa<ValueSequence>()) {
    const auto &value_seq = value->cast<ValueSequencePtr>()->value();
    InputType ret_type = grad_type;
    for (const auto &v : value_seq) {
      auto ret = SetValueGradInfo(v, grad_type);
      if (IsParam(ret)) {
        ret_type = ret;
      }
    }
    return ret_type;
  }
  if (value->isa<tensor::COOTensor>()) {
    const auto &coo_tensor = value->cast<tensor::COOTensorPtr>();
    const auto &indices_tensor = coo_tensor->GetIndices();
    return SetValueGradInfo(indices_tensor, grad_type);
  }
  if (value->isa<tensor::CSRTensor>()) {
    const auto &csr_tensor = value->cast<tensor::CSRTensorPtr>();
    const auto &indices_tensor = csr_tensor->GetIndices();
    return SetValueGradInfo(indices_tensor, grad_type);
  }
  if (value->isa<ValueDictionary>()) {
    const auto &dic_v = value->cast<ValueDictionaryPtr>()->value();
    for (const auto &v : dic_v) {
      (void)SetValueGradInfo(v.second, grad_type);
    }
  }
  return grad_type;
}

InputType AutoGradUtil::SetTensorGradInfo(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  auto auto_grad_meta_data = autograd::impl::GetAutogradMetaImpl(tensor);
  if (auto_grad_meta_data != nullptr) {
    if (auto_grad_meta_data->input_type() == InputType::kOpOutput) {
      return auto_grad_meta_data->input_type();
    }
    MS_LOG(DEBUG) << "Set input type for tensor " << tensor->id();
  } else if (tensor->is_parameter()) {
    MS_LOG(DEBUG) << "Create new auto grad meta for tensor " << tensor->id();
    auto_grad_meta_data = std::make_shared<AutoGradMetaData>();
    tensor->set_auto_grad_meta_data(auto_grad_meta_data);
    if (IsParamRequiresGrad(tensor)) {
      auto fn =
        std::make_shared<autograd::LeafNode>(tensor->param_info()->name(), tensor, tensor->shape(), tensor->Dtype());
      auto_grad_meta_data->set_requires_grad(true);
      auto_grad_meta_data->set_grad_node(fn);
    }
  }
  // Set weight tensor grad type
  if (tensor->is_parameter()) {
    auto_grad_meta_data->set_input_type(InputType::kParameter);
    return InputType::kParameter;
  }
  if (auto_grad_meta_data != nullptr && auto_grad_meta_data->input_type() == InputType::kInput) {
    return InputType::kInput;
  }
  return InputType::kConstant;
}

bool AutoGradUtil::IsPrimNeedGrad(const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  return kGradBlackList.find(prim->name()) == kGradBlackList.end();
}

ValuePtr AutoGradUtil::BaseRefToValue(const BaseRef &value, bool requires_grad, bool is_out_sequence) {
  MS_EXCEPTION_IF_NULL(value);
  ValuePtr ret;
  if (utils::isa<tensor::TensorPtr>(value)) {
    auto t = utils::cast<tensor::TensorPtr>(value);
    if (requires_grad) {
      t->set_auto_grad_meta_data(std::make_shared<AutoGradMetaData>(InputType::kOpOutput));
    }
    ret = t;
  } else if (utils::isa<ValuePtr>(value)) {
    ret = utils::cast<ValuePtr>(value);
  } else if (utils::isa<VectorRef>(value)) {
    auto vec_ref = utils::cast<VectorRef>(value);
    ret = VectorRefToValue(vec_ref, requires_grad, is_out_sequence);
  } else if (utils::isa<int>(value)) {
    ret = MakeValue(utils::cast<int>(value));
  } else if (utils::isa<float>(value)) {
    ret = MakeValue(utils::cast<float>(value));
  } else if (utils::isa<double>(value)) {
    ret = MakeValue(utils::cast<double>(value));
  } else if (utils::isa<bool>(value)) {
    ret = MakeValue(utils::cast<bool>(value));
  } else {
    MS_LOG(EXCEPTION) << "value is not support type " << value.ToString();
  }
  return ret;
}

ValuePtr AutoGradUtil::VectorRefToValue(const VectorRef &vec_ref, bool requires_grad, bool is_out_sequence) {
  MS_EXCEPTION_IF_NULL(vec_ref);
  size_t value_size = vec_ref.size();
  if (value_size == 1 && !is_out_sequence) {
    return BaseRefToValue(vec_ref[0], requires_grad, is_out_sequence);
  }
  std::vector<ValuePtr> v_list(value_size);
  for (size_t i = 0; i < value_size; ++i) {
    v_list[i] = BaseRefToValue(vec_ref[i], requires_grad, is_out_sequence);
  }
  return std::make_shared<ValueTuple>(v_list);
}

void AutoGradUtil::BuildViewAutoGradMeta(const tensor::TensorPtr &src_tensor, const tensor::TensorPtr &output,
                                         autograd::CreationType creation_type, bool requires_grad) {
  MS_EXCEPTION_IF_NULL(output);
  auto view_meta = autograd::impl::GetViewAutogradMetaImpl(src_tensor);
  autograd::ViewAutoGradMetaDataPtr cur_view_meta;
  if (view_meta != nullptr) {
    output->set_version(src_tensor->version());
    cur_view_meta = std::make_shared<autograd::ViewAutoGradMetaData>(
      view_meta->view_info().Union(), requires_grad ? InputType::kOpOutput : InputType::kUnkown,
      creation_type != autograd::CreationType::kDefault ? creation_type : view_meta->creation_type());
    output->set_auto_grad_meta_data(cur_view_meta);
  } else {
    if (src_tensor->auto_grad_meta_data() == nullptr) {
      // If base tensor is input of view op, we need construct auto_grad_meta_data for base tensor, to
      // avoid view tensor being inplaced by inplace op, which will need update grad info of base tensor.
      // we need construct auto_grad_meta_data in second thread rather than bprop thread.
      MS_LOG(DEBUG) << "Create new auto grad meta for input tensor of view op " << src_tensor->id();
      auto auto_grad_meta_data = std::make_shared<AutoGradMetaData>();
      src_tensor->set_auto_grad_meta_data(auto_grad_meta_data);
      if (IsParamRequiresGrad(src_tensor) && autograd::impl::GetUnsafeGradNodeImpl(src_tensor) == nullptr) {
        auto fn = std::make_shared<autograd::LeafNode>(src_tensor->param_info()->name(), src_tensor,
                                                       src_tensor->shape(), src_tensor->Dtype());
        auto_grad_meta_data->set_requires_grad(true);
        auto_grad_meta_data->set_grad_node(fn);
      }
    }
    // Temp method to avoid view tensor hold by grad.
    auto base_tensor = std::make_shared<tensor::Tensor>(*src_tensor);
    if (src_tensor->is_parameter()) {
      base_tensor->set_param_info(src_tensor->param_info());
    }
    base_tensor->set_storage_info(src_tensor->storage_info());
    base_tensor->set_device_address(nullptr);
    ViewInfo view_info(base_tensor);
    output->set_version(src_tensor->version());
    cur_view_meta = std::make_shared<autograd::ViewAutoGradMetaData>(
      std::move(view_info), requires_grad ? InputType::kOpOutput : InputType::kUnkown, creation_type);
    output->set_auto_grad_meta_data(cur_view_meta);
  }
  if (!requires_grad) {
    PyNativeAlgo::PyBoost::UpdateVersionAsync(cur_view_meta, output->version());
  }
}

void AutoGradUtil::SetInferOutputToGrad(const PyboostOpRunInfoPtr &op_run_info, const kernel::pyboost::OpPtr &op) {
  if (op->output_value_simple_info() != nullptr) {
    op_run_info->output_value_simple_info = op->output_value_simple_info();
    op_run_info->output_value_simple_info->is_tuple_output_ = false;
  }
}

void AutoGradUtil::SetInferOutputToGrad(const OpGradInfoPtr &op_grad_info, const kernel::pyboost::OpPtr &op) {
  if (op->output_value_simple_info() != nullptr) {
    op_grad_info->output_value_simple_info = op->output_value_simple_info();
    op_grad_info->output_value_simple_info->is_tuple_output_ = false;
  }
}

void AutoGradUtil::SetInferMultiOutputToGrad(const OpGradInfoPtr &op_grad_info, const kernel::pyboost::OpPtr &op) {
  if (op->output_value_simple_info() != nullptr) {
    op_grad_info->output_value_simple_info = op->output_value_simple_info();
    op_grad_info->output_value_simple_info->is_tuple_output_ = true;
  }
}

ValuePtr AutoGradUtil::MakeOutput(bool requires_grad, const kernel::pyboost::OpPtr &op,
                                  const tensor::TensorPtr &base_view) {
  return MakeOutput(requires_grad, op->outputs()[0], base_view);
}

ValuePtr AutoGradUtil::MakeMultiOutput(bool requires_grad, const kernel::pyboost::OpPtr &op,
                                       const tensor::TensorPtr &base_view) {
  return MakeOutput(requires_grad, op->outputs(), base_view);
}

ValuePtr AutoGradUtil::MakeMultiOutput(bool requires_grad, const kernel::pyboost::OpPtr &op,
                                       const ValueTuplePtr &base_view) {
  size_t size = op->outputs().size();
  std::vector<ValuePtr> output_values(size);
  auto inputs = base_view->value();
  if (inputs.size() != size) {
    MS_LOG(EXCEPTION) << "For multi inputs and multi outputs view op, inputs size should be same as outputs!";
  }
  for (size_t i = 0; i < size; ++i) {
    const auto &output_tensor = op->outputs()[i];
    MS_EXCEPTION_IF_NULL(output_tensor);
    const auto input_tensor = inputs[i]->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(input_tensor);
    // Set auto grad meta data for op output
    if (input_tensor != nullptr && output_tensor->storage_info() != nullptr) {
      BuildViewAutoGradMeta(input_tensor, output_tensor, autograd::CreationType::kDefault, requires_grad);
    } else if (requires_grad) {
      if (op->outputs()[i]->auto_grad_meta_data() == nullptr) {
        op->outputs()[i]->set_auto_grad_meta_data(std::make_shared<AutoGradMetaData>(InputType::kOpOutput));
      }
    }
    output_values[i] = output_tensor;
  }
  return std::make_shared<ValueTuple>(output_values);
}

ValuePtr AutoGradUtil::MakeOutput(bool requires_grad, const tensor::TensorPtr &output_tensor,
                                  const tensor::TensorPtr &base_view) {
  // delete NoneTypeNode check.
  if (base_view != nullptr && output_tensor->storage_info() != nullptr) {
    autograd::CreationType creationType =
      requires_grad ? autograd::CreationType::kDefault : autograd::CreationType::kNoGradMode;
    BuildViewAutoGradMeta(base_view, output_tensor, creationType, requires_grad);
  } else if (requires_grad) {
    if (output_tensor->auto_grad_meta_data() == nullptr) {
      output_tensor->set_auto_grad_meta_data(std::make_shared<AutoGradMetaData>(InputType::kOpOutput));
    } else {
      // View op from no grad mode has not input type, we need set it by inplace op,
      // which only worked in view inplace process.
      output_tensor->auto_grad_meta_data()->set_input_type(InputType::kOpOutput);
    }
  }
  return output_tensor;
}

ValuePtr AutoGradUtil::MakeOutput(bool requires_grad, const std::vector<tensor::TensorPtr> &output_tensors,
                                  const tensor::TensorPtr &base_view) {
  size_t size = output_tensors.size();
  std::vector<ValuePtr> output_values(size);
  for (size_t i = 0; i < size; ++i) {
    const auto &output_tensor = output_tensors[i];
    MS_EXCEPTION_IF_NULL(output_tensor);
    // Set auto grad meta data for op outputs
    if (base_view != nullptr && output_tensor->storage_info() != nullptr) {
      BuildViewAutoGradMeta(base_view, output_tensor, autograd::CreationType::kMultiOutput, requires_grad);
    } else if (requires_grad) {
      if (output_tensors[i]->auto_grad_meta_data() == nullptr) {
        output_tensors[i]->set_auto_grad_meta_data(std::make_shared<AutoGradMetaData>(InputType::kOpOutput));
      } else {
        output_tensors[0]->auto_grad_meta_data()->set_input_type(InputType::kOpOutput);
      }
    }
    output_values[i] = output_tensor;
  }
  return std::make_shared<ValueTuple>(output_values);
}

void AutoGradUtil::BumpVersion(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  auto tensor = value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor);
  tensor->BumpVersion();
}

bool AutoGradUtil::NeedGrad(const tensor::TensorPtr &input_tensor) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  if (input_tensor->is_parameter()) {
    return IsParamRequiresGrad(input_tensor);
  }
  auto grad_meta = input_tensor->auto_grad_meta_data();
  return grad_meta != nullptr && grad_meta->requires_grad();
}

bool AutoGradUtil::NeedGrad(const std::vector<ValuePtr> &input_values) {
  for (const ValuePtr &input_arg : input_values) {
    MS_EXCEPTION_IF_NULL(input_arg);
    if (input_arg->isa<tensor::Tensor>()) {
      const auto input_tensor = input_arg->cast<tensor::TensorPtr>();
      if (NeedGrad(input_tensor)) {
        return true;
      }
    } else if (input_arg->isa<ValueSequence>()) {
      auto value_seq = input_arg->cast<ValueSequencePtr>()->value();
      if (NeedGrad(value_seq)) {
        return true;
      }
    } else if (input_arg->isa<tensor::COOTensor>() || input_arg->isa<tensor::CSRTensor>()) {
      return true;
    } else if (input_arg->isa<ValueDictionary>()) {
      auto dict_val = input_arg->cast<ValueDictionaryPtr>()->value();
      for (auto kv : dict_val) {
        if (NeedGrad({kv.second})) {
          return true;
        }
      }
    }
    MS_LOG(DEBUG) << "Get value " << input_arg->ToString();
  }
  return false;
}

ValuePtr AutoGradUtil::BuildSpecialValueGrad(const ValuePtr &value, const tensor::TensorPtr &grad,
                                             autograd::FuncBuilder *func_builder, const SpecialType &type) {
  MS_EXCEPTION_IF_NULL(value);
  if (grad != nullptr) {
    return grad;
  }
  if (value->isa<tensor::Tensor>()) {
    const auto tensor = value->cast<tensor::TensorPtr>();
    if (EnableDispatch()) {
      NativeFunc::set_device_target(tensor->device_type());
      MS_LOG(DEBUG) << "[Dispatch]Change device target to " << device::GetDeviceNameByType(tensor->device_type());
    }
    return (type == SpecialType::kZerosLikeType ? func_builder->Zeros(tensor) : func_builder->Ones(tensor));
  }
  if (value->isa<ValueSequence>()) {
    ValuePtr zero_value = nullptr;
    auto v_seq = value->cast<ValueSequencePtr>();
    ValuePtrList v_list;
    for (const auto &item : v_seq->value()) {
      (void)v_list.emplace_back(BuildSpecialValueGrad(item, grad, func_builder, type));
    }
    return std::make_shared<ValueTuple>(v_list);
  }
  if (value->isa<Scalar>()) {
    auto fake_tensor = tensor::from_scalar(0, value->type());
    return BuildSpecialValueGrad(fake_tensor, grad, func_builder, type);
  }
  if (value->isa<tensor::CSRTensor>()) {
    auto csr_tensor = value->cast<tensor::CSRTensorPtr>();
    return WrapCSRTensor(csr_tensor, BuildSpecialValueGrad(csr_tensor->GetValues(), grad, func_builder, type));
  }
  if (value->isa<tensor::COOTensor>()) {
    auto coo_tensor = value->cast<tensor::COOTensorPtr>();
    return WrapCOOTensor(coo_tensor, BuildSpecialValueGrad(coo_tensor->GetValues(), grad, func_builder, type));
  }
  MS_LOG(INFO) << "For value " << value->ToString() << ", the type is not tensor or scalar";
  return tensor::from_scalar(0);
}

namespace {
std::string GenerateCacheKeyWithGradInfo(const FuncGraphPtr &call_graph, const std::string &cache_key) {
  if (common::GetCompileConfig("GRAD_JIT_FILTER") != "1") {
    return cache_key;
  }
  constexpr auto need_grad_hash = "need_grad_hash";
  const auto &call_graph_attrs = call_graph->attrs();
  auto need_hash_iter = call_graph_attrs.find(need_grad_hash);
  if (need_hash_iter == call_graph_attrs.end()) {
    MS_LOG(WARNING) << "Failed to find need_grad_hash for graph " << call_graph->ToString();
    return cache_key;
  }
  return cache_key + "_" + std::to_string(GetValue<size_t>(need_hash_iter->second));
}
}  // namespace

CallBackFn AutoGradUtil::CreateGraphCallBack(const FuncGraphPtr &call_graph, const std::string &cache_key,
                                             const GraphCallCondition &graph_call_condition) {
  // kFlagJitCallGraph is set true to avoid compilig call_graph whe compiling the main graph
  call_graph->set_flag(kFlagJitCallGraph, true);
  // call graph not inline to grad top
  call_graph->set_flag(FUNC_GRAPH_FLAG_NO_INLINE, true);
  // Pynative bprop graph flag
  call_graph->set_flag(kFlagIsPynativeBpropGraph, true);
  pipeline::ResourcePtr resource;
  constexpr auto kNeedCompile = "NeedCompile";

  const std::string &cache_key_with_grad_info = GenerateCacheKeyWithGradInfo(call_graph, cache_key);
  MS_LOG(INFO) << "cache_key_with_grad_info: " << cache_key_with_grad_info;
  const auto it = jit_call_graph_compile_cache_.find(cache_key_with_grad_info);
  bool need_compile = (it == jit_call_graph_compile_cache_.end());
  if (need_compile) {
    resource = std::make_shared<pipeline::Resource>();
    resource->set_func_graph(call_graph);
    if (graph_call_condition.is_func_grad_) {
      auto manager = resource->manager();
      manager->AddFuncGraph(call_graph, false);
      (void)opt::EnvironConversion(resource);
      if (graph_call_condition.jit_out_has_dict_) {
        MS_LOG(DEBUG) << "Jit out is dict, need convert make dict to pyexecute";
        (void)mindspore::opt::RewriterAfterOptA(resource->func_graph(), resource);
      }
    }
    if (graph_call_condition.is_jit_graph_) {
      (void)jit_call_graph_compile_cache_.emplace(cache_key_with_grad_info, resource);
    }
    resource->SetResult(kNeedCompile, true);
  } else {
    resource = it->second;
    // If resource func graph not compile(not call run grad graph), but hit cache
    need_compile = resource->GetResult(kNeedCompile).cast<bool>();
  }
  MS_EXCEPTION_IF_NULL(resource);
  bool is_control_flow = graph_call_condition.is_control_flow_;
  auto fn = [resource, need_compile, is_control_flow, kNeedCompile](const VectorRef &arg_list) -> VectorRef {
    if (need_compile) {
      MS_LOG(DEBUG) << "Start emit action for graph " << resource->func_graph()->ToString();
      auto manager = resource->manager();
      manager->AddFuncGraph(resource->func_graph(), true);
      // kFlagJitCallGraph is set false to compile sub graph in control flow
      if (is_control_flow) {
        for (const auto &g : manager->func_graphs()) {
          g->set_flag(kFlagJitCallGraph, false);
        }
      }
      pipeline::JitCompilingScope jit_compiling_scope;
      (void)TaskEmitAction(resource);
      (void)ExecuteAction(resource);
      resource->SetResult(kNeedCompile, false);
    }
    MS_LOG(DEBUG) << "Start execute action for graph " << resource->func_graph()->ToString();
    pipeline::JitRunningScope jit_running_scope;
    VectorRef outputs;
    if (common::AnfAlgo::IsGraphOutputValueNodeOrParameter(resource->func_graph()->output(), arg_list, &outputs)) {
      return outputs;
    }
    VmEvalPtr run = resource->GetResult(pipeline::kOutput).cast<VmEvalPtr>();
    return utils::cast<VectorRef>((*run)(arg_list));
  };
  return fn;
}

void AutoGradUtil::CreateHighOrderGraph(const FuncGraphPtr &func_graph, const VectorRef &input_args,
                                        const VectorRef &out, const std::string &cache_key) {
  MS_LOG(DEBUG) << "Begin create high order graph";
  auto op_run_info = std::make_shared<FrontendOpRunInfo>();
  op_run_info->requires_grad = true;
  auto input_value = AutoGradUtil::VectorRefToValue(input_args, false, true)->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(input_value);
  op_run_info->op_grad_info->input_value = input_value->value();
  op_run_info->input_size = op_run_info->op_grad_info->input_value.size();
  MS_EXCEPTION_IF_NULL(out);
  auto out_value = AutoGradUtil::BaseRefToValue(out, true, true);
  // Get output values
  if (!out_value->isa<ValueSequence>()) {
    std::vector<ValuePtr> out_v{out_value};
    out_value = std::make_shared<ValueTuple>(out_v);
  }
  auto first_grad_fg = func_graph;
  // Get input values
  PyNativeAlgo::Common::SetGraphInputAndWeightsInfo(op_run_info, first_grad_fg);
  (void)first_grad_fg->transforms().erase(kGrad);
  op_run_info->op_grad_info->out_value = out_value;
  op_run_info->op_grad_info->out_abs = first_grad_fg->output()->abstract();
  auto resource = std::make_shared<pipeline::Resource>();
  auto opt = opt::Optimizer::MakeEmptyOptimizer(resource);
  opt->set_is_first_order_j(false);
  resource->set_func_graph(first_grad_fg);
  py::gil_scoped_acquire gil;
  first_grad_fg = pipeline::HighGradBpropGraphPass(resource);
  auto grad_graph = ad::Grad(first_grad_fg, opt);
  MS_EXCEPTION_IF_NULL(grad_graph);
  MS_LOG(INFO) << "Finish using adgrad generate second order graph of graph: " << first_grad_fg->ToString();
  auto grad_param = std::make_shared<GradParam>(op_run_info->op_grad_info);
  grad_param->fg = grad_graph;
  grad_param->source_fg = first_grad_fg;
  grad_param->is_control_flow = true;
  // Add flag to avoid high order miss cache.
  grad_param->graph_cache_key = cache_key + "_Grad";
  runtime::Pipeline::Get().WaitBpropStage();
  if (!autograd::KPynativeWithFProp(grad_param)) {
    MS_LOG(EXCEPTION) << "Failed to make adjoint for jit cnode";
  }
}

PrimitivePyPtr AutoGradUtil::BuildBpropCutPrim(const PrimitivePtr &prim, bool is_need_recompute) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_py = prim->cast<PrimitivePyPtr>();
  MS_EXCEPTION_IF_NULL(prim_py);
  auto bprop_cut = std::make_shared<PrimitivePy>("bprop_cut");
  bprop_cut->CopyHookFunction(prim_py);
  prim_py->AddBpropCutPrim(bprop_cut);
  if (prim->HasAttr("cell_id")) {
    auto cell_id = GetValue<std::string>(prim->GetAttr("cell_id"));
    if (!cell_id.empty()) {
      (void)bprop_cut->AddAttr("cell_hook", MakeValue(true));
      (void)bprop_cut->AddAttr("cell_id", MakeValue(cell_id));
    }
  }
  // Only custom op need add this attr, hook function not need.
  if (prim->HasAttr("custom_op_bprop")) {
    (void)bprop_cut->AddAttr("custom_op_bprop", MakeValue(true));
  }
  (void)bprop_cut->AddAttr("custom_op_name", MakeValue(prim->name()));
  if (is_need_recompute) {
    (void)bprop_cut->AddAttr("is_recompute", MakeValue(true));
  }
  return bprop_cut;
}

void AutoGradUtil::CheckRecomputeInputs(const ValuePtrList &inputs, bool is_need_recompute) {
  if (!is_need_recompute) {
    return;
  }
  for (const auto &input : inputs) {
    if (!input->isa<ValueSequence>()) {
      continue;
    }
    const auto &seq = input->cast<ValueSequencePtr>();
    const auto val = seq->value();
    if (NeedGrad(val)) {
      MS_LOG(EXCEPTION) << "For recompute cell, now we do not support calculate tensor's gradient from tuple. "
                           "You need check your inputs of construct function from recompute cell, and not put "
                           "tensors in tuple which need grad!";
    }
  }
}

void AutoGradUtil::ClearAutoGradStaticCache() { jit_call_graph_compile_cache_.clear(); }

void AutoGradUtil::CheckAndSetAbstract(const OpGradInfoPtr &op_grad_info) {
  MS_EXCEPTION_IF_NULL(op_grad_info);
  if (op_grad_info->output_value_simple_info != nullptr) {
    MS_LOG(DEBUG) << "Convert op " << op_grad_info->op_prim->name() << " simple infer info to abstract";
    ConvertSimpleInferInfoToAbstract(op_grad_info);
    return;
  }

  // View op input abs and output abs maybe nullptr
  if (MS_UNLIKELY(op_grad_info->input_abs.empty())) {
    // Get inputs abstract
    MS_LOG(DEBUG) << "Op " << op_grad_info->op_prim->name() << " inputs abstract not set, set it now";
    for (const auto &v : op_grad_info->input_value) {
      // For use abstract cache on tensor
      op_grad_info->input_abs.emplace_back(kGradAbstractConverter.ConvertAbstract(v));
    }
  }
  if (op_grad_info->out_abs == nullptr) {
    op_grad_info->out_abs = kGradAbstractConverter.ConvertAbstract(op_grad_info->out_value);
  }
}

void AutoGradUtil::CacheOutputAbstract(const ValuePtr &v, const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(v);
  MS_EXCEPTION_IF_NULL(abs);

  // Just check size.
  if (v->isa<ValueSequence>()) {
    const auto &value_seq = v->cast<ValueSequencePtr>();
    const auto &abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    if (abs_seq == nullptr) {
      MS_LOG(EXCEPTION) << "Abstract is not abstract sequence, get " << abs->ToString();
    }
    size_t value_size = value_seq->size();
    if (value_size != abs_seq->size()) {
      MS_LOG(EXCEPTION) << "Abstract size " << abs_seq->size() << " is not equal to value size " << value_size;
    }
  }
}

void AutoGradUtil::CheckAndCloneInplaceInput(const kernel::pyboost::OpPtr &inplace_op, const PrimitivePtr &prim,
                                             device::DeviceType device_target, ValuePtrList &&inputs) {
  auto input_tensor = inputs[0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  ValuePtr val = nullptr;
  if (!kernel::pyboost::OpRunStatus::Get().RequireGrad() ||
      !BpropExpander::IsCloneInplaceInput(BpropCallback(prim, &inputs, &val))) {
    return;
  }
  MS_LOG(DEBUG) << "Begin clone src value for op " << prim->name();
  kernel::pyboost::OpRunStatus::Get().set_run_info(kernel::pyboost::OpStatus(true, device_target));
  auto output = kernel::pyboost::clone(input_tensor);
  inplace_op->set_clone_tensor(output);
}

ValuePtr AutoGradUtil::ShallowCopyAndDetach(const ValuePtr &value) {
  if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    auto copy_tensor = std::make_shared<tensor::Tensor>(*tensor);
    copy_tensor->set_auto_grad_meta_data(nullptr);
    return copy_tensor;
  } else if (value->isa<ValueSequence>()) {
    auto val_seq = value->cast<ValueSequencePtr>();
    std::vector<ValuePtr> res;
    for (const auto &val : val_seq->value()) {
      (void)res.emplace_back(ShallowCopyAndDetach(val));
    }
    return std::make_shared<ValueTuple>(res);
  }
  return value;
}

TensorPtr AutoGradUtil::ViewAsSelfWithNoGrad(const TensorPtr &self) {
  kernel::pyboost::OpStatus status{false, DeviceManagerConf::GetInstance()->device_type()};
  kernel::pyboost::OpRunStatus::Get().set_run_info(std::move(status));
  kernel::pyboost::RequireGradGuard require_grad_guard(false);
  return kernel::pyboost::view(self, self->shape());
}

TensorPtr AutoGradUtil::Add(const TensorPtr &input, const TensorPtr &other) {
  kernel::pyboost::OpStatus status{false, DeviceManagerConf::GetInstance()->device_type()};
  kernel::pyboost::OpRunStatus::Get().set_run_info(std::move(status));
  return kernel::pyboost::add(input, other);
}

TensorPtr AutoGradUtil::Clone(const TensorPtr &input) {
  kernel::pyboost::OpStatus status{false, DeviceManagerConf::GetInstance()->device_type()};
  kernel::pyboost::OpRunStatus::Get().set_run_info(std::move(status));
  return kernel::pyboost::clone(input);
}

std::vector<autograd::TensorMeta> AutoGradUtil::GenerateInputsMeta(const std::vector<autograd::Edge> &inputs) {
  std::vector<autograd::TensorMeta> inputs_meta;
  inputs_meta.reserve(inputs.size());
  for (const auto &input : inputs) {
    if (!input.is_defined()) {
      (void)inputs_meta.emplace_back();
      continue;
    }
    MS_EXCEPTION_IF_CHECK_FAIL(input.input_index < input.grad_node->mutable_metadata().size(),
                               "index should less than metadata size!");
    (void)inputs_meta.emplace_back(input.grad_node->mutable_metadata()[input.input_index]);
  }
  return inputs_meta;
}

ValuePtrList AutoGradUtil::AutoCastAndReduce(const ValuePtrList &gradients,
                                             const std::vector<autograd::TensorMeta> &inputs_meta) {
  ValuePtrList grads;
  grads.reserve(gradients.size());
  if (gradients.size() < inputs_meta.size()) {
    MS_LOG(EXCEPTION) << "Grad size should lager than forward inputs, but got " << gradients.size() << " vs "
                      << inputs_meta.size();
  }
  for (size_t i = 0; i < inputs_meta.size(); ++i) {
    const auto &input_info = inputs_meta[i];
    if (input_info.is_default() || gradients[i]->isa<None>()) {
      (void)grads.emplace_back(gradients[i]);
      continue;
    }
    MS_EXCEPTION_IF_NULL(gradients[i]);
    auto grad_tensor = gradients[i]->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(grad_tensor);
    if (input_info.IsSameShape(grad_tensor->shape())) {
      (void)grads.emplace_back(Cast(grad_tensor, input_info.dtype()));
      continue;
    }
    if (!input_info.IsBroadcastTo(grad_tensor->shape())) {
      MS_LOG(EXCEPTION)
        << "For grad interface or custom function, grad tensor should be broadcast to expected shape, but got "
        << grad_tensor->shape() << " vs " << input_info.shape();
    }
    grad_tensor = Cast(ReduceGrad(grad_tensor, input_info.shape()), input_info.dtype());
    (void)grads.emplace_back(grad_tensor);
  }
  return grads;
}

tensor::TensorPtr AutoGradUtil::ReduceGrad(const tensor::TensorPtr &grad, const std::vector<int64_t> &reduce_shape) {
  kernel::pyboost::OpStatus status{false, DeviceManagerConf::GetInstance()->device_type()};
  kernel::pyboost::OpRunStatus::Get().set_run_info(std::move(status));
  auto src_size = reduce_shape.size();
  auto grad_size = grad->shape().size();
  auto keep_axis = std::make_shared<BoolImm>(false);
  std::vector<ValuePtr> reduce_axis;
  reduce_axis.reserve(grad_size);
  if (src_size == 0) {
    std::vector<ValuePtr> axes;
    return kernel::pyboost::sum_ext(grad, std::make_shared<ValueTuple>(axes), keep_axis, std::nullopt);
  }
  size_t expanded_axis = grad_size - src_size;
  for (size_t i = 0; i < expanded_axis; ++i) {
    (void)reduce_axis.emplace_back(std::make_shared<Int64Imm>(i));
  }
  for (size_t i = expanded_axis; i < grad_size; ++i) {
    if (grad->shape()[i] != 1 && reduce_shape[i - expanded_axis] == 1) {
      (void)reduce_axis.emplace_back(std::make_shared<Int64Imm>(i));
    }
  }
  return kernel::pyboost::sum_ext(grad, std::make_shared<ValueTuple>(reduce_axis), keep_axis, std::nullopt);
}

tensor::TensorPtr AutoGradUtil::Cast(const tensor::TensorPtr &grad, const TypePtr &cast_dtype) {
  if (grad->data_type() != cast_dtype->type_id()) {
    MS_LOG(DEBUG) << "grad dtype is not same as input, try to cast dtype";

    kernel::pyboost::OpStatus status{false, DeviceManagerConf::GetInstance()->device_type()};
    kernel::pyboost::OpRunStatus::Get().set_run_info(std::move(status));
    return kernel::pyboost::cast(grad, std::make_shared<Int64Imm>(static_cast<int64_t>(cast_dtype->type_id())));
  }
  return grad;
}

bool BpropCallback::IsNotRequiresGrad(size_t index) const {
  // Check Tensor need grad.
  runtime::Pipeline::Get().WaitBpropStage();
  return !AutoGradUtil::NeedGrad({(*inputs_)[index]});
}

void BpropCallback::FreeDeviceAddress(ValuePtr *value) const {
  *value = PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(*value);
}

AutoGradGuard::AutoGradGuard(bool require_grad) {
  origin_require_grad_ = kernel::pyboost::OpRunStatus::Get().RequireGrad();
  origin_enable_grad_ = PyNativeExecutor::GetInstance()->enable_grad();
  kernel::pyboost::OpRunStatus::Get().SetRequireGrad(require_grad);
  PyNativeExecutor::GetInstance()->set_enable_grad(require_grad);
}

AutoGradGuard::~AutoGradGuard() {
  kernel::pyboost::OpRunStatus::Get().ResetRequireGrad(origin_require_grad_);
  PyNativeExecutor::GetInstance()->set_enable_grad(origin_enable_grad_);
}
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_GRAD_UTILS_H_
