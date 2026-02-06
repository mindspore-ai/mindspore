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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inner_index.h"
#include "ir/tensor_new.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
std::vector<TensorPtr> ConvertEmptyTensor(const ValueTuplePtr &tuple, device::DeviceType device_type) {
  // It is temporarily used: when the shape is 9 zeros, similar to ":" in x[(1,2,..), :, (..),].
  std::vector<TensorPtr> result;
  const auto &values = tuple->value();
  for (const auto &value : values) {
    auto tensor = GetValue<TensorPtr>(value);
    auto shape = tensor->shape();
    constexpr auto kSize9 = 9;
    if (shape.size() == kSize9 && std::all_of(shape.begin(), shape.end(), [](int i) { return i == 0; })) {
      auto type_id = tensor->data_type();
      std::vector<int64_t> empty_shape({0});
      result.push_back(tensor::from_spec(type_id, empty_shape, device_type));
    } else {
      result.push_back(tensor);
    }
  }
  return result;
}
}  // namespace

tensor::TensorPtr InnerIndexAscendCustomize(const TensorPtr &input_tensor, const ValueTuplePtr &indices_tensor_list) {
  MS_LOG(DEBUG) << "InnerIndex Ascend start";
  auto output = InferOutput<1>(prim::kPrimInnerIndex, input_tensor, indices_tensor_list);

  // Process shape of 9 zeros
  std::vector<TensorPtr> indices_tensor_vector =
    ConvertEmptyTensor(indices_tensor_list, input_tensor->device_address()->GetDeviceType());

  // Set address
  TensorToDevice(input_tensor, indices_tensor_vector);
  TensorAllocate(output);

  DISPATCH_ACLNN(aclnnIndex, input_tensor, indices_tensor_vector, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
