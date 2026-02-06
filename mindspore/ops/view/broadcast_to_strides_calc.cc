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

#include "view/broadcast_to_strides_calc.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "ops/op_def_utils.h"
#include "utils/check_convert_utils.h"
#include "primitive/array_op_name.h"

namespace mindspore::ops {
namespace {
inline static void BroadcastToCheck(const std::string &prim_name, const std::vector<int64_t> &proposed_shape,
                                    const std::vector<int64_t> &x_shape) {
  MS_CHECK_VALUE(
    x_shape.size() <= proposed_shape.size(),
    "For primitive [BroadcastTo]: input's rank should be less equal to the number of proposed_shape, but got " +
      std::to_string(x_shape.size()) + " and " + std::to_string(proposed_shape.size()));

  auto outer_dim_offset = proposed_shape.size() - x_shape.size();
  bool has_infer_dim = true;
  if (proposed_shape.end() == find(proposed_shape.begin(), proposed_shape.end(), -1)) {
    has_infer_dim = false;
  }
  if (has_infer_dim) {
    for (size_t i = 0; i < proposed_shape.size(); i++) {
      if (proposed_shape[i] == -1) {
        if (i < outer_dim_offset) {
          MS_EXCEPTION(ValueError) << "For '" << prim_name
                                   << "', -1 in init shape is in an incompatible "
                                      "location with given input tensor, -1 index in init shape: "
                                   << i << " but -1 can only be in index" << x_shape.size()
                                   << "onwards for this input.";
        }
      }
    }
  }
  for (size_t i = 0; i < x_shape.size(); i++) {
    if (proposed_shape[i + outer_dim_offset] == -1) {
      continue;
    }
    if (proposed_shape[i + outer_dim_offset] != x_shape[i] && x_shape[i] != 1) {
      MS_EXCEPTION(ValueError)
        << "For '" << prim_name
        << "', in order to broadcast, each dimension pair must be equal or input dimension is 1 or target "
           "dimension is -1. But got x_shape: "
        << ShapeVectorToStr(x_shape) << ", target shape: " << ShapeVectorToStr(proposed_shape) << ".";
    }
  }
}
}  // namespace

TensorStorageInfoPtrList BroadCastToStrideCalc(const std::vector<int64_t> &old_shape,
                                               const std::vector<int64_t> &old_strides,
                                               const TensorStorageInfoPtr &storage_info,
                                               const std::vector<int64_t> &proposed_shape) {
  MS_LOG(DEBUG) << "BroadCastTo: input shape " << old_shape << ", input stride " << old_strides << ", storage_info "
                << (storage_info != nullptr ? storage_info->ToString() : "null") << ", shape " << proposed_shape;
  BroadcastToCheck(kBroadcastToOpName, proposed_shape, old_shape);
  auto [ori_shape, ori_strides, old_storage_offset] = GetOriShapeStridesAndOffset(old_shape, old_strides, storage_info);
  int64_t ndim = SizeToInt(proposed_shape.size());
  int64_t tensor_ndim = SizeToInt(old_shape.size());
  std::vector<int64_t> new_strides(ndim);
  if (MS_UNLIKELY(tensor_ndim == 0)) {
    bool is_contiguous = IsContiguous(proposed_shape, new_strides);
    auto new_storage_info =
      std::make_shared<TensorStorageInfo>(proposed_shape, std::move(new_strides), old_storage_offset,
                                          std::move(ori_shape), std::move(ori_strides), is_contiguous);
    MS_LOG(DEBUG) << "BroadCastTo: output storage_info " << new_storage_info->ToString();
    return {new_storage_info};
  }

  std::vector<int64_t> new_shape(ndim);
  for (int64_t i = ndim - 1; i >= 0; --i) {
    int64_t offset = ndim - 1 - i;
    int64_t dim = tensor_ndim - 1 - offset;
    auto size = (dim >= 0) ? old_shape[dim] : 1;
    auto stride = (dim >= 0) ? old_strides[dim] : new_shape[i + 1] * new_strides[i + 1];
    auto target_size = proposed_shape[i];
    if (target_size == -1) {
      target_size = size;
    }
    if (size != target_size) {
      size = target_size;
      stride = 0;
    }
    new_shape[i] = size;
    new_strides[i] = stride;
  }
  bool is_contiguous = IsContiguous(new_shape, new_strides);
  auto new_storage_info =
    std::make_shared<TensorStorageInfo>(std::move(new_shape), std::move(new_strides), old_storage_offset,
                                        std::move(ori_shape), std::move(ori_strides), is_contiguous);
  MS_LOG(DEBUG) << "BroadCastTo: output storage_info " << new_storage_info->ToString();
  return {new_storage_info};
}

TensorStorageInfoPtrList BroadcastToBasicTypeCalc(const mindspore::tensor::TensorPtr &input_tensor,
                                                  const std::vector<int64_t> &shape) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  return BroadCastToStrideCalc(input_tensor->shape(), input_tensor->stride(), input_tensor->storage_info(), shape);
}

TensorStorageInfoPtrList BroadcastToCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (!inputs[kInputIndex0]->isa<tensor::Tensor>()) {
    return {};
  }
  auto input_tensor = inputs[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto proposed_shape = GetValue<std::vector<int64_t>>(inputs[kInputIndex1]);
  return BroadcastToBasicTypeCalc(input_tensor, proposed_shape);
}

REG_VIEW_STRIDES_CALC_FUN(BroadcastTo, BroadcastToCalc);
}  // namespace mindspore::ops
