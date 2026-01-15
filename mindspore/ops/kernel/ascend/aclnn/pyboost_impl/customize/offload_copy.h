/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_OPS_KERNEL_ASCEND_ACLNN_PYBOOST_IMPL_CUSTOMIZE_OFFLOAD_COPY_H_
#define MINDSPORE_MINDSPORE_CCSRC_OPS_KERNEL_ASCEND_ACLNN_PYBOOST_IMPL_CUSTOMIZE_OFFLOAD_COPY_H_

#include <memory>
#include "ir/tensor.h"
#include "kernel/ascend/visible.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

// D2H copy: copy device tensor to host (CPU) tensor with pinned memory
// @param stream_id: the stream to execute the copy on (typically copy stream)
// @param device_tensor: source tensor on device
// @return: new host tensor with pinned memory containing the copied data
tensor::TensorPtr OPS_ASCEND_API CopyToHost(size_t stream_id, const tensor::TensorPtr &device_tensor);

// H2D copy: copy host tensor to device tensor
// @param stream_id: the stream to execute the copy on (typically copy stream)
// @param host_tensor: source tensor on host (CPU)
// @return: new device tensor containing the copied data
tensor::TensorPtr OPS_ASCEND_API CopyToDevice(size_t stream_id, const tensor::TensorPtr &host_tensor);

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_OPS_KERNEL_ASCEND_ACLNN_PYBOOST_IMPL_CUSTOMIZE_OFFLOAD_COPY_H_
