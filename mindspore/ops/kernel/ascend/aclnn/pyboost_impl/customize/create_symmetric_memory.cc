/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/create_symmetric_memory.h"

#include <memory>
#include <string>
#include <vector>
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
ShapeVector GetShape(const ValueTuplePtr &shape) {
  ShapeVector output_shape;
  for (size_t i = 0; i < shape->size(); ++i) {
    int64_t shape_i = std::static_pointer_cast<Int64Imm>((*shape)[i])->value();
    output_shape.push_back(shape_i);
  }
  return output_shape;
}

tensor::TensorPtr CreateSymmetricMemoryAscendCustomize(const std::shared_ptr<OpRunner> &op, const ValueTuplePtr &shape,
                                                       const std::optional<Int64ImmPtr> &dtype,
                                                       const StringImmPtr &group) {
  MS_LOG(DEBUG) << "Call CreateSymmetricMemory start";
  TypeId data_type = static_cast<TypeId>(GetValue<int64_t>(dtype.value()));

  auto device_ctx = runtime::OpRunner::GetDeviceContext(device::DeviceType::kAscend);
  MS_EXCEPTION_IF_NULL(device_ctx);

  ShapeVector output_shape = GetShape(shape);
  std::vector<tensor::TensorPtr> outputs;
  PyBoostUtils::CreateOutputTensor(data_type, output_shape, &outputs);
  PyBoostUtils::PrepareOpOutputs(device_ctx, op->stream_id(), outputs);
  for (auto output : outputs) {
    auto device_address = std::static_pointer_cast<device::DeviceAddress>(output->device_address());
    device_address->set_allocator(device_ctx->device_res_manager_->symmetric_memory_allocator());
  }

  op->set_outputs(outputs);

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, device_ctx]() {
    const auto &outputs = op->outputs();
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_ctx, outputs);
    auto address = std::dynamic_pointer_cast<device::DeviceAddress>(outputs[0]->device_address());
    MS_LOG(DEBUG) << "Run device task CreateSymmetricMemory end, address: " << address->GetPtr();
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
