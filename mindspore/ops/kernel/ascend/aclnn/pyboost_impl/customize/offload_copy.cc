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

#include "kernel/ascend/aclnn/pyboost_impl/customize/offload_copy.h"
#include "kernel/ascend/aclnn/pyboost_impl/customize/inplace_copy.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/pynative/utils/pyboost/offload_copy_registry.h"
#include "include/pynative/utils/pyboost/functions/composite/empty_like.h"
#include "utils/convert_utils_base.h"
#include "utils/current_state.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_i.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
class CurrentStreamGuard {
 public:
  explicit CurrentStreamGuard(size_t stream_id) : old_stream_id_(CurrentStream::id()) {
    CurrentStream::set_id(SizeToUint(stream_id));
  }

  ~CurrentStreamGuard() { CurrentStream::set_id(old_stream_id_); }

 private:
  uint32_t old_stream_id_;
};

tensor::TensorPtr InplaceCopyOnStream(size_t stream_id, const tensor::TensorPtr &dst, const tensor::TensorPtr &src,
                                      const BoolImmPtr &non_blocking) {
  CurrentStreamGuard guard(stream_id);
  return InplaceCopyAscendCustomize(dst, src, non_blocking);
}
}  // namespace

tensor::TensorPtr CopyToHost(size_t stream_id, const tensor::TensorPtr &device_tensor) {
  MS_LOG(DEBUG) << "CopyToHost start, stream_id=" << stream_id;
  MS_EXCEPTION_IF_NULL(device_tensor);

  // Step 1: Create output tensor on CPU with pinned memory using empty_like
  auto cpu_device = std::make_shared<Int64Imm>(static_cast<int64_t>(DEVICE_CPU));
  auto pin_memory = std::make_shared<BoolImm>(true);
  auto dst_tensor = empty_like(device_tensor, std::nullopt, cpu_device, pin_memory);
  MS_EXCEPTION_IF_NULL(dst_tensor);

  bool skip_tracker = ProfileTracker::ProfileTrackerTask(prim::kPrimInplaceCopy);

  // Step 2: Call InplaceCopyAscendCustomize (dst=host CPU, src=device Ascend) -> D2H
  auto non_blocking = std::make_shared<BoolImm>(true);
  InplaceCopyOnStream(stream_id, dst_tensor, device_tensor, non_blocking);
  ProfileTracker::ProfileTrackerInput(prim::kPrimInplaceCopy, skip_tracker, dst_tensor, device_tensor, non_blocking);
  ProfileTracker::ProfileTrackerOutput(prim::kPrimInplaceCopy, skip_tracker, dst_tensor);
  MS_LOG(DEBUG) << "CopyToHost end, output shape=" << dst_tensor->shape();
  return dst_tensor;
}

tensor::TensorPtr CopyToDevice(size_t stream_id, const tensor::TensorPtr &host_tensor) {
  MS_LOG(DEBUG) << "CopyToDevice start, stream_id=" << stream_id;
  MS_EXCEPTION_IF_NULL(host_tensor);

  // Step 1: Create output tensor on Ascend device using empty_like
  auto ascend_device = std::make_shared<Int64Imm>(static_cast<int64_t>(DEVICE_ASCEND));
  auto pin_memory = std::make_shared<BoolImm>(false);
  auto dst_tensor = empty_like(host_tensor, std::nullopt, ascend_device, pin_memory);
  MS_EXCEPTION_IF_NULL(dst_tensor);

  bool skip_tracker = ProfileTracker::ProfileTrackerTask(prim::kPrimInplaceCopy);
  // Step 2: Call InplaceCopyAscendCustomize (dst=device Ascend, src=host CPU) -> H2D
  auto non_blocking = std::make_shared<BoolImm>(true);
  InplaceCopyOnStream(stream_id, dst_tensor, host_tensor, non_blocking);
  ProfileTracker::ProfileTrackerInput(prim::kPrimInplaceCopy, skip_tracker, dst_tensor, host_tensor, non_blocking);
  ProfileTracker::ProfileTrackerOutput(prim::kPrimInplaceCopy, skip_tracker, dst_tensor);
  MS_LOG(DEBUG) << "CopyToDevice end, output shape=" << dst_tensor->shape();
  return dst_tensor;
}

// Static registration: register implementations when mindspore_ops_ascend.so is loaded
static OffloadCopyRegistrar g_offload_copy_registrar(CopyToHost, CopyToDevice);

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
