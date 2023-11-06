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

#include "plugin/device/ascend/kernel/pyboost/customize/conv2d.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/py_boost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::vector<int64_t> ExpandVector(const std::vector<int64_t> &value, const size_t &expected_dim) {
  if (value.size() == 1) {
    std::vector<int64_t> expand_vec;
    for (size_t i = 0; i < expected_dim; i++) {
      expand_vec.emplace_back(value[0]);
    }
    return expand_vec;
  }
  return value;
}

tensor::TensorPtr Conv2DAscendCall(const PrimitivePtr &primitive, const device::DeviceContext *device_context,
                                   const tensor::TensorPtr &input_tensor, const tensor::TensorPtr &weight_tensor,
                                   const std::optional<tensor::TensorPtr> &bias_tensor,
                                   const std::vector<int64_t> &stride, const std::vector<int64_t> &padding,
                                   const std::vector<int64_t> &dilation, const int64_t &groups,
                                   const std::vector<tensor::TensorPtr> &outputs) {
  MS_LOG(DEBUG) << "Call start";
  const size_t dim = 2;
  const auto &expand_stride = ExpandVector(stride, dim);
  const auto &expand_padding = ExpandVector(padding, dim);
  const auto &expand_dilation = ExpandVector(dilation, dim);
  auto transposed = false;
  std::vector<int64_t> output_padding = {0, 0};
  auto stream_ptr = device_context->device_res_manager_->GetStream(kDefaultStreamIndex);
  if (outputs.empty()) {
    MS_LOG(EXCEPTION) << "outputs is empty";
    return nullptr;
  }
  auto cube_math_type = GetCubeMathType();
  EXEC_NPU_CMD(aclnnConvolution, stream_ptr, input_tensor, weight_tensor, bias_tensor, expand_stride, expand_padding,
               expand_dilation, transposed, output_padding, groups, outputs[0], cube_math_type);
  MS_LOG(DEBUG) << "Launch end";
  return outputs[0];
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
