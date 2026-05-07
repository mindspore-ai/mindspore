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
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/runtime/op_runner.h"
#include "utils/current_state.h"
#include "acl/acl_rt.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"

namespace mindspore::kernel::pyboost {
namespace {
constexpr size_t kAlignSize = 512;
}

tensor::TensorPtr AllFiniteAscendCustomize(const ValueTuplePtr &tensors_tensor_list) {
  MS_EXCEPTION_IF_NULL(tensors_tensor_list);
  auto op =
    std::make_shared<OpRunner>(prim::kPrimAllFinite, runtime::OpRunner::GetDeviceContext(device::DeviceType::kAscend));
  op->set_stream_id(CurrentStream::id());
  MS_LOG(DEBUG) << "Start AllFinite ascend customize";
  OpRunner::InferOpOutput(op, tensors_tensor_list);
  // ValueTuple to std::vector
  std::vector<TensorPtr> tensors_tensor_list_vector = ConvertValueTupleToVector<TensorPtr>(tensors_tensor_list);
  auto device_context = op->device_context();
  TensorToDevice(op, tensors_tensor_list_vector);
  TensorMalloc(op, op->outputs());

  const auto &output = op->output(0);
  auto stream_ptr = device_context->device_res_manager_->GetStream(op->stream_id());
  MS_EXCEPTION_IF_NULL(stream_ptr);
  // Wrap in DispatchRun so GetMutablePtr() is called after async allocation completes
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([output, stream_ptr]() {
    runtime::OpExecutor::DispatchLaunchTask(
      [output_device_ptr = output->device_address()->GetMutablePtr(), stream_ptr]() {
        auto ret = CALL_ASCEND_API(aclrtMemsetAsync, output_device_ptr, kAlignSize, 0, kAlignSize, stream_ptr);
        if (ret != ACL_SUCCESS) {
          MS_LOG(EXCEPTION) << "Call runtime aclrtMemsetAsync error, ret[" << ret << "]";
        }
      });
  }));

  for (size_t i = 0; i < tensors_tensor_list_vector.size(); i++) {
    DISPATCH_LAUNCH_ACLNN(aclnnAllFinite, tensors_tensor_list_vector[i], output);
  }
  return output;
}
}  // namespace mindspore::kernel::pyboost
