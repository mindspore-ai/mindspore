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

#include "kernel/ascend/aclnn/pyboost_impl/customize/index.h"
#include "kernel/ascend/aclnn/pyboost_impl/auto_generate/inner_non_zero.h"
#include "kernel/ascend/aclnn/pyboost_impl/auto_generate/inner_index.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
std::vector<TensorPtr> IndexGetNewTensor(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                         const std::vector<TensorPtr> &tensors) {
  kernel::pyboost::RequireGradGuard require_grad_guard(false);
  std::vector<TensorPtr> result{};
  auto input_shape = input_tensor->shape();
  if (input_shape.size() == 0) {
    MS_EXCEPTION(ValueError) << "For 'Index', too many indices for tensor of dimension " << input_shape.size();
  }
  if (tensors.size() > input_shape.size()) {
    MS_EXCEPTION(ValueError) << "For 'Index', too many indices for tensor of dimension " << input_shape.size()
                             << " (got " << tensors.size() << ")";
  }
  bool needCast = false;
  TypeId indicesDtype = tensors[0]->data_type();
  for (const auto &tensor : tensors) {
    auto type_id = tensor->data_type();
    if (type_id != kNumberTypeInt64 && type_id != kNumberTypeInt32 && type_id != kNumberTypeBool &&
        type_id != kNumberTypeUInt8) {
      MS_EXCEPTION(TypeError) << "For 'Index', tensors used as indices must be long, int, uint8, or bool tensors";
    }
    if (type_id == kNumberTypeBool || type_id == kNumberTypeUInt8) {
      auto shape = tensor->shape();
      auto rank = SizeToLong(shape.size());
      for (int64_t j = 0; j < rank; j++) {
        auto srcIdx = result.size() + j;
        if (shape[j] != input_shape[srcIdx]) {
          MS_EXCEPTION(ValueError) << "For 'Index', the shape of the mask " << tensor->ElementsNum() << " at index "
                                   << j << " does not match the shape of the indexed tensor " << input_shape
                                   << " at index " << srcIdx;
        }
      }
      auto nonzero_op = CREATE_PYBOOST_OP(InnerNonZero, device::DeviceType::kAscend);
      auto nonzero_tensor = nonzero_op->Call(tensor);
      for (int64_t j = 0; j < rank; j++) {
        auto select_tensor = select_ext_view(nonzero_tensor, kIndex0, j);
        result.emplace_back(select_tensor);
      }
    } else {
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

tensor::TensorPtr IndexAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                       const ValueTuplePtr &indices_tensor_list) {
  MS_LOG(DEBUG) << "Index Ascend start";
  std::vector<TensorPtr> indices_tensor_vector = ConvertValueTupleToVector<TensorPtr>(indices_tensor_list);
  if (indices_tensor_vector.size() == 0) {
    MS_EXCEPTION(ValueError) << "For 'Index', 'indices' shape can't be empty.";
  }
  auto new_indices_tensor_vector = IndexGetNewTensor(op, input_tensor, indices_tensor_vector);
  ValueTuplePtr new_indices_tensor_list = PyBoostUtils::ConvertTensorVectorToTuple(new_indices_tensor_vector);

  auto inner_index_op = CREATE_PYBOOST_OP(InnerIndex, device::DeviceType::kAscend);
  auto index_out = inner_index_op->Call(input_tensor, new_indices_tensor_list);
  op->set_outputs(inner_index_op->outputs());
  MS_LOG(DEBUG) << "Index Ascend end";
  return index_out;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
