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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_COMM_COMMON_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_COMM_COMMON_H_

#include <vector>
#include <memory>
#include <string>
#include "ir/tensor.h"
#include "include/pynative/utils/pyboost/op_runner.h"
#include "hccl/hccl.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void CommonCommRunTask(const std::function<void(void)> &run_func);
void CommonCommAscendFunc(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor, const StringImmPtr &group,
                          const std::function<void(const HcclComm &, void *)> &launch_func,
                          const std::function<void(const DeviceEventPtr &, size_t)> &post_func, int64_t rank = -1);

// Get device mutable ptr from tensor
void *GetDevicePtrFromTensor(const std::string &op_name, const tensor::TensorPtr &tensor);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_COMM_COMMON_H_
