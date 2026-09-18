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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inplace_index_put.h"
#include <functional>
#include "kernel/ascend/aclnn/pyboost_impl/auto_generate/inner_inplace_index_put.h"
#include "kernel/ascend/aclnn/pyboost_impl/auto_generate/inner_non_zero.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/op_register.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
// Expands a bool or uint8 mask into the integer indices that select its elements.
void AppendInplaceIndexPutMaskIndexes(const TensorPtr &tensor, const std::vector<TensorPtr> &tensors,
                                      const std::vector<int64_t> &input_shape, std::vector<TensorPtr> *result) {
  auto shape = tensor->shape();
  auto rank = SizeToLong(shape.size());
  if (rank == 0) {
    // A 0-dim mask selects the whole tensor (True) or nothing (False) and consumes no dimension of the input.
    // It is only forwarded to aclnnIndexPutImpl when it is the sole index; combined with other indices it
    // carries no coordinate information, so it is ignored and does not take part in the dimension count.
    if (tensors.size() == 1) {
      result->emplace_back(tensor);
    }
    return;
  }
  // A mask consumes 'rank' dimensions, so check the accumulated dims before expanding it to avoid reading
  // input_shape out of bounds below.
  if (result->size() + LongToSize(rank) > input_shape.size()) {
    MS_EXCEPTION(ValueError) << "For 'InplaceIndexPut', too many indices for tensor of dimension " << input_shape.size()
                             << " (got " << (result->size() + LongToSize(rank)) << ")";
  }
  for (int64_t j = 0; j < rank; j++) {
    auto srcIdx = result->size() + j;
    if (shape[j] != input_shape[srcIdx]) {
      MS_EXCEPTION(ValueError) << "For 'InplaceIndexPut', the shape of the mask " << tensor->ElementsNum()
                               << " at index " << j << " does not match the shape of the indexed tensor " << input_shape
                               << " at index " << srcIdx;
    }
  }
  // Normalize masks to integer coordinates before dispatching to aclnnIndexPutImpl.
  auto nonzero_op = CREATE_PYBOOST_OP(InnerNonZero, device::DeviceType::kAscend);
  auto nonzero_tensor = nonzero_op->Call(tensor);
  for (int64_t j = 0; j < rank; j++) {
    result->emplace_back(select_ext_view(nonzero_tensor, kIndex0, j));
  }
}

std::vector<TensorPtr> InplaceIndexGetNewTensor(const TensorPtr &input_tensor, const std::vector<TensorPtr> &tensors) {
  kernel::pyboost::RequireGradGuard require_grad_guard(false);
  std::vector<TensorPtr> result{};
  auto input_shape = input_tensor->shape();
  if (input_shape.size() == 0) {
    MS_EXCEPTION(ValueError) << "For 'InplaceIndexPut', too many indices for tensor of dimension "
                             << input_shape.size();
  }
  bool needCast = false;
  TypeId indicesDtype = tensors[0]->data_type();
  for (const auto &tensor : tensors) {
    auto type_id = tensor->data_type();
    if (type_id != kNumberTypeInt64 && type_id != kNumberTypeInt32 && type_id != kNumberTypeBool &&
        type_id != kNumberTypeUInt8) {
      MS_EXCEPTION(TypeError)
        << "For 'InplaceIndexPut', tensors used as indices must be long, int, uint8, or bool tensors";
    }
    if (type_id == kNumberTypeBool || type_id == kNumberTypeUInt8) {
      AppendInplaceIndexPutMaskIndexes(tensor, tensors, input_shape, &result);
    } else {
      // An integer index consumes one dimension of the input, so check the accumulated dims here as well: the
      // number of index tensors is not the number of consumed dimensions once 0-dim masks are involved.
      if (result.size() + 1 > input_shape.size()) {
        MS_EXCEPTION(ValueError) << "For 'InplaceIndexPut', too many indices for tensor of dimension "
                                 << input_shape.size() << " (got " << (result.size() + 1) << ")";
      }
      result.emplace_back(tensor);
    }
    if (indicesDtype != type_id) {
      needCast = true;
    }
  }
  if (needCast) {
    for (size_t i = 0; i < result.size(); i++) {
      if (result[i]->data_type() == kNumberTypeInt32) {
        result[i] = PyBoostUtils::CastTensor(result[i], kNumberTypeInt64, device::DeviceType::kAscend);
      }
    }
  }
  return result;
}
}  // namespace

tensor::TensorPtr InplaceIndexPutAscendCustomize(const tensor::TensorPtr &input_tensor,
                                                 const ValueTuplePtr &indices_tensor_list,
                                                 const tensor::TensorPtr &values_tensor, const BoolImmPtr &accumulate) {
  MS_LOG(DEBUG) << "InplaceIndexPut Ascend start";
  const auto &input_shape = input_tensor->shape();
  const auto &values_shape = values_tensor->shape();
  std::vector<TensorPtr> indices_tensor_vector = ConvertValueTupleToVector<TensorPtr>(indices_tensor_list);
  auto input_numel =
    std::accumulate(input_shape.begin(), input_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  auto values_numel =
    std::accumulate(values_shape.begin(), values_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  if (input_numel == 0 || values_numel == 0 || indices_tensor_vector.size() == 0) {
    return input_tensor;
  }
  auto new_indices_tensor_vector = InplaceIndexGetNewTensor(input_tensor, indices_tensor_vector);

  ValueTuplePtr new_indices_tensor_list = PyBoostUtils::ConvertTensorVectorToTuple(new_indices_tensor_vector);

  auto inner_inp_index_put_op = CREATE_PYBOOST_OP(InnerInplaceIndexPut, device::DeviceType::kAscend);
  auto index_out = inner_inp_index_put_op->Call(input_tensor, new_indices_tensor_list, values_tensor, accumulate);
  MS_LOG(DEBUG) << "InplaceIndexPut Ascend end";
  return index_out;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
