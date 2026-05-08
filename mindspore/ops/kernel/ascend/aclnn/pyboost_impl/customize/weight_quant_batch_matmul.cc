/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/weight_quant_batch_matmul.h"
#include <memory>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "primitive/op_name.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "device_address/device_address_maker.h"
#include "ir/tensor_new.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_w.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
std::vector<int64_t> GetWeightQuantBatchMatmulPerm(const TensorPtr &weight_tensor) {
  const auto &shape = weight_tensor->shape();
  int64_t size = shape.size();
  std::vector<int64_t> perm(size);
  if (size < ops::kSize2) {
    perm[0] = 0;
    return perm;
  }
  perm[size - 1] = size - SizeToLong(kDim2);
  perm[size - kDim2] = size - 1;
  for (int64_t i = 0; i < size - ops::kSize2; ++i) {
    perm[i] = i;
  }
  return perm;
}
}  // namespace
tensor::TensorPtr WeightQuantBatchMatmulV2AscendCustomize(
  const tensor::TensorPtr &x_tensor, const tensor::TensorPtr &weight_tensor,
  const tensor::TensorPtr &antiquant_scale_tensor, const std::optional<tensor::TensorPtr> &antiquant_offset_tensor,
  const std::optional<tensor::TensorPtr> &quant_scale_tensor,
  const std::optional<tensor::TensorPtr> &quant_offset_tensor, const std::optional<tensor::TensorPtr> &bias_tensor,
  const BoolImmPtr &transpose_x, const BoolImmPtr &transpose_weight, const Int64ImmPtr &antiquant_group_size) {
  auto output = InferOutput<1>(prim::kPrimWeightQuantBatchMatmul, x_tensor, weight_tensor, antiquant_scale_tensor,
                               antiquant_offset_tensor, quant_scale_tensor, quant_offset_tensor, bias_tensor,
                               transpose_x, transpose_weight, antiquant_group_size);
  auto transpose_x_imm = GetValue<bool>(transpose_x);
  auto transpose_weight_imm = GetValue<bool>(transpose_weight);
  auto antiquant_group_size_imm = GetValue<int64_t>(antiquant_group_size);

  TensorPtr new_weight_tensor = weight_tensor;
  TypeId weight_dtype = weight_tensor->data_type();
  if (weight_dtype == kNumberTypeInt4) {
    ShapeVector weight_shape = weight_tensor->shape();
    int kInt4ShapeMul = 2;
    weight_shape.back() *= kInt4ShapeMul;
    const ShapeVector &new_weight_shape = weight_shape;
    new_weight_tensor =
      std::make_shared<tensor::Tensor>(weight_tensor->data_type(), new_weight_shape, weight_tensor->device_address());
  }

  TensorToDevice(x_tensor, new_weight_tensor, antiquant_scale_tensor, antiquant_offset_tensor, quant_scale_tensor,
                 quant_offset_tensor, bias_tensor);
  TensorMalloc(output);

  if (weight_dtype == kNumberTypeInt4) {
    auto device_address = std::static_pointer_cast<device::DeviceAddress>(new_weight_tensor->device_address());
    size_t int4_real_size = device_address->GetSize() / kSizeTwo;
    device_address->SetSize(int4_real_size);
  }

  TensorPtr x_tensor_trans = x_tensor;
  kernel::pyboost::RequireGradGuard require_grad_guard(false);
  if (transpose_x_imm) {
    x_tensor_trans = transpose(x_tensor_trans, GetWeightQuantBatchMatmulPerm(x_tensor_trans));
  }
  TensorPtr weight_tensor_trans = new_weight_tensor;
  if (transpose_weight_imm) {
    weight_tensor_trans = transpose(weight_tensor_trans, GetWeightQuantBatchMatmulPerm(weight_tensor_trans));
  }
  DISPATCH_ACLNN(aclnnWeightQuantBatchMatmulV2, x_tensor_trans, weight_tensor_trans, antiquant_scale_tensor,
                 antiquant_offset_tensor, quant_scale_tensor, quant_offset_tensor, bias_tensor,
                 antiquant_group_size_imm, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
