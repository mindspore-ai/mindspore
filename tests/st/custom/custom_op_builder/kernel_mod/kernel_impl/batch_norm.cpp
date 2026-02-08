/**
* Copyright 2025 Huawei Technologies Co., Ltd
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
#include <vector>
#include <map>
#include <string>
#include <utility>
#include "ops/ops_func_impl/op_func_impl.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"
#include "custom_op_api.h"
#include "module.h"

namespace my_custom_ops {
using namespace mindspore;
using namespace mindspore::kernel;
using namespace mindspore::device::ascend;
using namespace mindspore::ops;

class OPS_API BatchNormOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    const auto &input_shape = input_infos[kIndex0]->GetShape();
    auto channel = input_shape[kIndex1];
    const std::vector<int64_t> save_mv_shape{channel};
    return {input_shape, save_mv_shape, save_mv_shape};
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    const auto &input_type = input_infos[kIndex0]->GetType();
    const auto &weight_info = input_infos[kIndex1];
    auto save_mv_type = weight_info->IsNone() ? input_type : weight_info->GetType();
    return {input_type, save_mv_type, save_mv_type};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class BatchNormExtAscend : public AclnnKernelMod {
 public:
  BatchNormExtAscend() : AclnnKernelMod(std::move("aclnnBatchNorm")) {}

  ~BatchNormExtAscend() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(stream_ptr);
    RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex4],
          training_, momentum_, eps_, outputs[kIndex0], outputs[kIndex1], outputs[kIndex2]);
    return true;
  }

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    training_ = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex5]);

    auto eps_dtype_id = inputs[kIndex7]->dtype_id();
    eps_ = (eps_dtype_id == kNumberTypeFloat32) ? static_cast<double>(inputs[kIndex7]->GetValueWithCheck<float>())
                                                : inputs[kIndex7]->GetValueWithCheck<double>();

    auto momentum_dtype_id = inputs[kIndex6]->dtype_id();
    momentum_ = (momentum_dtype_id == kNumberTypeFloat32)
                  ? static_cast<double>(inputs[kIndex6]->GetValueWithCheck<float>())
                  : inputs[kIndex7]->GetValueWithCheck<double>();

    GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex4],
                          training_, momentum_, eps_, outputs[kIndex0], outputs[kIndex1], outputs[kIndex2]);
  }

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE()

  bool training_;
  double momentum_;
  double eps_;
};

}  // namespace my_custom_ops

REG_GRAPH_MODE_OP(batch_norm, my_custom_ops::BatchNormOpFuncImpl, my_custom_ops::BatchNormExtAscend);