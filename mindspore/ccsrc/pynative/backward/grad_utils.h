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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_GRAD_GRAD_UTILS_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_GRAD_GRAD_UTILS_H_

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "pynative/utils/base.h"
#include "include/pynative/utils/pyboost/op_runner.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/forward/pyboost/forward_task.h"
#include "pynative/backward/op_grad/func_builder.h"
#include "include/frontend/operator/primitive_py.h"
#include "include/utils/pynative/variable.h"

namespace mindspore {
namespace pynative {
using CallBackFn = std::function<VectorRef(const VectorRef &arg_list)>;
enum class SpecialType { kZerosLikeType = 0, kOnesLikeType = 1 };

class BpropCallback final : public expander::bprop::PynativeCallback {
 public:
  BpropCallback(const PrimitivePtr &prim, ValuePtrList *inputs, ValuePtr *output)
      : prim_(prim), inputs_(inputs), output_(output) {}
  const std::string &opname() const override { return prim_->name(); }
  ValuePtr *GetInput(size_t index) const override { return &(*inputs_)[index]; }
  ValuePtrList *GetInputs() const override { return inputs_; }
  ValuePtr *GetOutput() const override { return output_; }
  bool IsNotRequiresGrad(size_t index) const override;
  void FreeDeviceAddress(ValuePtr *value) const override;

 private:
  const PrimitivePtr &prim_;
  ValuePtrList *inputs_;
  ValuePtr *output_;
};

class AutoGradGuard {
 public:
  explicit AutoGradGuard(bool require_grad);
  ~AutoGradGuard();

 private:
  bool origin_require_grad_{false};
  bool origin_enable_grad_{false};
};

struct AutoGradUtil {
  // Common grad function
  static InputType SetValueGradInfo(const ValuePtr &value, InputType grad_type);
  static InputType SetTensorGradInfo(const tensor::TensorPtr &tensor);
  static ValuePtr BaseRefToValue(const BaseRef &value, bool requires_grad, bool is_out_sequence);
  static ValuePtr VectorRefToValue(const VectorRef &vec_ref, bool requires_grad, bool is_out_sequence);
  static void BuildViewAutoGradMeta(const tensor::TensorPtr &src_tensor, const tensor::TensorPtr &output,
                                    autograd::CreationType creation_type, bool requires_grad);
  static void SetInferOutputToGrad(const PyboostOpRunInfoPtr &op_run_info, const kernel::pyboost::OpPtr &op);
  static void SetInferOutputToGrad(const OpGradInfoPtr &op_grad_info, const kernel::pyboost::OpPtr &op);
  static void SetInferMultiOutputToGrad(const OpGradInfoPtr &op_grad_info, const kernel::pyboost::OpPtr &op);
  static ValuePtr MakeOutput(bool requires_grad, const kernel::pyboost::OpPtr &op,
                             const tensor::TensorPtr &base_view = nullptr);
  static ValuePtr MakeMultiOutput(bool requires_grad, const kernel::pyboost::OpPtr &op,
                                  const tensor::TensorPtr &view_base = nullptr);
  // Multi inputs and multi outputs view op enter here, temp code need discard.
  static ValuePtr MakeMultiOutput(bool requires_grad, const kernel::pyboost::OpPtr &op, const ValueTuplePtr &base_view);
  static ValuePtr MakeOutput(bool requires_grad, const tensor::TensorPtr &output_tensor,
                             const tensor::TensorPtr &base_view = nullptr);
  static ValuePtr MakeOutput(bool requires_grad, const std::vector<tensor::TensorPtr> &output_tensors,
                             const tensor::TensorPtr &base_view = nullptr);
  static void BumpVersion(const ValuePtr &value);

  static bool IsPrimNeedGrad(const PrimitivePtr &prim);
  static bool NeedGrad(const tensor::TensorPtr &input_tensor);
  static bool NeedGrad(const std::vector<ValuePtr> &input_values);
  static ValuePtr BuildSpecialValueGrad(const ValuePtr &value, const tensor::TensorPtr &grad,
                                        autograd::FuncBuilder *func_builder, const SpecialType &type);
  static inline bool IsParam(InputType grad_type) {
    return grad_type == InputType::kParameter || grad_type == InputType::kInput;
  }
  static inline bool IsParamRequiresGrad(const tensor::TensorPtr &tensor) {
    return tensor->param_info() != nullptr && tensor->param_info()->requires_grad();
  }
  static CallBackFn CreateGraphCallBack(const FuncGraphPtr &call_graph, const std::string &cache_key,
                                        const GraphCallCondition &graph_call_condition);
  static void CreateHighOrderGraph(const FuncGraphPtr &first_grad_fg, const VectorRef &input_args, const VectorRef &out,
                                   const std::string &cache_key);
  static PrimitivePyPtr BuildBpropCutPrim(const PrimitivePtr &prim, bool is_need_recompute = false);
  static void CheckRecomputeInputs(const ValuePtrList &inputs, bool is_need_recompute);
  static void ClearAutoGradStaticCache();
  static void CheckAndSetAbstract(const OpGradInfoPtr &op_grad_info);
  static void CacheOutputAbstract(const ValuePtr &v, const abstract::AbstractBasePtr &abs);
  static void CheckAndCloneInplaceInput(const kernel::pyboost::OpPtr &inplace_op, const PrimitivePtr &prim,
                                        device::DeviceType device_target, ValuePtrList &&inputs);
  static ValuePtr ShallowCopyAndDetach(const ValuePtr &value);
  static TensorPtr ViewAsSelfWithNoGrad(const TensorPtr &self);
  static TensorPtr Add(const TensorPtr &input, const TensorPtr &other);
  static TensorPtr Clone(const TensorPtr &input);
  static std::vector<autograd::TensorMeta> GenerateInputsMeta(const std::vector<autograd::Edge> &inputs);
  static ValuePtrList AutoCastAndReduce(const ValuePtrList &gradients,
                                        const std::vector<autograd::TensorMeta> &inputs_meta);
  static tensor::TensorPtr ReduceGrad(const tensor::TensorPtr &grad, const std::vector<int64_t> &reduce_shape);
  static tensor::TensorPtr Cast(const tensor::TensorPtr &grad, const TypePtr &dtype);
};
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_GRAD_GRAD_UTILS_H_
