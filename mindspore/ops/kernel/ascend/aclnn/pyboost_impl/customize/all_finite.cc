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

#include "kernel/ascend/aclnn/pyboost_impl/customize/all_finite.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "acl/acl_rt.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore::kernel::pyboost {
namespace {
constexpr size_t kAlignSize = 512;
}

tensor::TensorPtr AllFiniteAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                           const ValueTuplePtr &tensors_tensor_list) {
  MS_EXCEPTION_IF_NULL(op);
  MS_EXCEPTION_IF_NULL(tensors_tensor_list);
  MS_LOG(DEBUG) << "Start AllFinite ascend customize";
  OpRunner::InferOpOutput(op, tensors_tensor_list);
  // ValueTuple to std::vector
  std::vector<TensorPtr> tensors_tensor_list_vector = ConvertValueTupleToVector<TensorPtr>(tensors_tensor_list);
  auto device_context = op->device_context();
  PyBoostUtils::PrepareOpInputs(device_context, op->stream_id(), tensors_tensor_list_vector);
  PyBoostUtils::PrepareOpOutputs(device_context, op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, tensors_tensor_list_vector]() {
    MS_LOG(DEBUG) << "Run device task AllFinite start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, tensors_tensor_list_vector);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);
    auto stream_ptr = device_context->device_res_manager_->GetStream(op->stream_id());
    MS_EXCEPTION_IF_NULL(stream_ptr);
    runtime::OpExecutor::DispatchLaunchTask(
      [output_device_ptr = outputs[0]->device_address()->GetMutablePtr(), stream_ptr]() {
        auto ret = CALL_ASCEND_API(aclrtMemsetAsync, output_device_ptr, kAlignSize, 0, kAlignSize, stream_ptr);
        if (ret != ACL_SUCCESS) {
          MS_LOG(EXCEPTION) << "Call runtime aclrtMemsetAsync error, ret[" << ret << "]";
        }
      });

    for (size_t i = 0; i < tensors_tensor_list_vector.size(); i++) {
      LAUNCH_ACLNN(aclnnAllFinite, device_context, op->stream_id(), tensors_tensor_list_vector[i], outputs[0]);
    }
    MS_LOG(DEBUG) << "Run device task AllFinite end";
  }));
  return op->output(0);
}
}  // namespace mindspore::kernel::pyboost
