//
// Created by jojo on 2023/10/30.
//

#include "plugin/device/ascend/kernel/pyboost/bmm_ascend.h"
#include <algorithm>
#include <functional>
#include <memory>
#include "ir/tensor.h"
#include "runtime/device/device_address_utils.h"
#include "runtime/hardware/device_context_manager.h"
#include "transform/acl_ir/op_api_exec.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr BmmAscend::Call(const tensor::TensorPtr &input, const tensor::TensorPtr &mat2) {
  MS_LOG(DEBUG) << "Call start";
  InferOutput(input, mat2);
  // Don't need to allocate memory for Scalar.
  DeviceMalloc(input, mat2);
  auto stream_ptr = device_context_->device_res_manager_->GetStream(kDefaultStreamIndex);
  LAUNCH_ACLNN_CUBE(aclnnBatchMatMul, stream_ptr, input, mat2, output(0));
  MS_LOG(DEBUG) << "Launch end";
  return outputs_[0];
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore