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

#include "kernel/ascend/aclnn/pyboost_impl/customize/quant_matmul.h"
#include <string>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_tensor_wrapper.h"
#include "kernel/ascend/acl_ir/acl_convert.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"
#include "primitive/auto_generate/gen_ops_primitive_q.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
int64_t CheckAndGetGroups(const std::vector<int64_t> &group_sizes_list) {
  int64_t groups = 0;
  if (group_sizes_list.empty()) {
    return groups;
  }
  constexpr size_t kGroupSizeLen = 3;
  if (group_sizes_list.size() != kGroupSizeLen) {
    MS_EXCEPTION(ValueError) << "For QuantMatmul, group_sizes should contain three elements, but got "
                             << group_sizes_list;
  }
  auto group_m = group_sizes_list[kIndex0];
  auto group_n = group_sizes_list[kIndex1];
  auto group_k = group_sizes_list[kIndex2];
  constexpr int64_t group_max = 65535LL;
  if (group_m > group_max || group_n > group_max || group_k > group_max) {
    MS_EXCEPTION(ValueError) << "For QuantMatmul, all elements of group_sizes can't be larger than 65535, but got "
                             << group_sizes_list;
  }
  if (group_m < 0 || group_n < 0 || group_k < 0) {
    MS_EXCEPTION(ValueError) << "For QuantMatmul, all elements of group_sizes can't be less than 0, but got "
                             << group_sizes_list;
  }
  constexpr auto kGroupmBitOffset = 32;
  constexpr auto kGroupnBitOffset = 16;

  groups = static_cast<int64_t>((static_cast<uint64_t>(group_m) << kGroupmBitOffset) +
                                (static_cast<uint64_t>(group_n) << kGroupnBitOffset) + static_cast<uint64_t>(group_k));
  return groups;
}

TensorPtr QuantMatmulCreateEmpty(const ShapeVector &shape, const TypeId &type) {
  auto tensor = std::make_shared<Tensor>(type, shape);
  TensorMalloc(tensor);
  return tensor;
}

aclDataType GetQuantMatmulAclDtype(const TensorPtr &tensor, const std::optional<Int64ImmPtr> &dtype_override,
                                   bool byte_mx_scale = false) {
  MS_EXCEPTION_IF_NULL(tensor);
  if (byte_mx_scale && !dtype_override.has_value() && tensor->data_type() == kNumberTypeUInt8) {
    return ACL_FLOAT8_E8M0;
  }
  const auto dtype =
    dtype_override.has_value() ? static_cast<TypeId>(dtype_override.value()->value()) : tensor->data_type();
  return device::ascend::AclConverter::ConvertType(dtype);
}
}  // namespace

tensor::TensorPtr QuantMatmulAscendCustomize(
  const TensorPtr &x1, const TensorPtr &x2, const TensorPtr &scale, const std::optional<TensorPtr> &offset,
  const std::optional<TensorPtr> &pertoken_scale, const std::optional<TensorPtr> &bias,
  const std::optional<Int64ImmPtr> &output_dtype, const std::optional<Int64ImmPtr> &x1_dtype,
  const std::optional<Int64ImmPtr> &x2_dtype, const std::optional<Int64ImmPtr> &pertoken_scale_dtype,
  const std::optional<Int64ImmPtr> &scale_dtype, const std::optional<ValueTuplePtr> &group_sizes) {
  auto output = InferOutput<1>(prim::kPrimQuantMatmul, x1, x2, scale, offset, pertoken_scale, bias, output_dtype,
                               x1_dtype, x2_dtype, pertoken_scale_dtype, scale_dtype, group_sizes);

  const device::ascend::AclTensorWrapper x1_wrapper = {x1, GetQuantMatmulAclDtype(x1, x1_dtype)};
  const device::ascend::AclTensorWrapper x2_wrapper = {x2, GetQuantMatmulAclDtype(x2, x2_dtype)};
  const device::ascend::AclTensorWrapper scale_wrapper = {scale, GetQuantMatmulAclDtype(scale, scale_dtype, true)};
  std::optional<device::ascend::AclTensorWrapper> pertoken_scale_wrapper;
  if (pertoken_scale.has_value()) {
    pertoken_scale_wrapper = device::ascend::AclTensorWrapper{
      pertoken_scale.value(), GetQuantMatmulAclDtype(pertoken_scale.value(), pertoken_scale_dtype, true)};
  }
  auto output_dtype_id = output->data_type();
  auto y_scale = QuantMatmulCreateEmpty({0}, output_dtype_id);
  auto x1_offset = QuantMatmulCreateEmpty({0}, output_dtype_id);
  auto y_offset = QuantMatmulCreateEmpty({0}, output_dtype_id);
  auto transpose1 = false;
  auto transpose2 = false;
  auto group_sizes_list = ConvertValueTupleToVector<int64_t>(group_sizes);
  auto group_size_val = CheckAndGetGroups(group_sizes_list);

  TensorToDevice(x1, x2, scale, offset, pertoken_scale, bias);
  TensorMalloc(output);

  MS_LOG(DEBUG) << "QuantMatmul x1.shape = " << ShapeVectorToStr(x1->shape())
                << ", dtype = " << TypeIdToString(x1->Dtype()->type_id())
                << "; x2.shape = " << ShapeVectorToStr(x2->shape())
                << ", dtype = " << TypeIdToString(x2->Dtype()->type_id())
                << "; scale.shape = " << ShapeVectorToStr(scale->shape())
                << ", dtype = " << TypeIdToString(scale->Dtype()->type_id());

  if (scale->data_type() == kNumberTypeFloat32 && !pertoken_scale.has_value() && output_dtype_id != kNumberTypeInt32) {
    // quant_param path
    auto scale_shape = scale->shape();
    auto quant_param_shape = scale_shape;
    if (quant_param_shape.size() == 1 && offset.has_value()) {
      auto offset_shape = offset.value()->shape();
      quant_param_shape = scale_shape.at(kIndex0) > offset_shape.at(kIndex0) ? scale_shape : offset_shape;
    }
    auto quant_param = QuantMatmulCreateEmpty(quant_param_shape, kNumberTypeInt64);
    DISPATCH_ACLNN(aclnnTransQuantParamV2, scale, offset, quant_param);
    DISPATCH_ACLNN(aclnnQuantMatmulV5, x1_wrapper, x2_wrapper, pertoken_scale, quant_param, y_scale, x1_offset, offset,
                   y_offset, bias, transpose1, transpose2, group_size_val, output);
  } else {
    DISPATCH_ACLNN(aclnnQuantMatmulV5, x1_wrapper, x2_wrapper, pertoken_scale_wrapper, scale_wrapper, y_scale,
                   x1_offset, offset, y_offset, bias, transpose1, transpose2, group_size_val, output);
  }
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
