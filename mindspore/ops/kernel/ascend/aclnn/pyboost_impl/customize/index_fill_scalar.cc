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

#include "kernel/ascend/aclnn/pyboost_impl/customize/index_fill_scalar.h"
#include <memory>
#include <string>

#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

tensor::TensorPtr IndexFillScalarAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input,
                                                 const Int64ImmPtr &dim, const TensorPtr &index,
                                                 const ScalarPtr &value) {
  OpRunner::InferOpOutput(op, input, dim, index, value);

  auto dim_imm = GetValue<int64_t>(dim);
  std::vector<int64_t> index_vector = {};
  auto index_cpu = index->cpu();
  TypeId tensor_type_id = static_cast<TypeId>(index_cpu->data_type_c());
  size_t elem_num = index_cpu->DataSize();
  if (tensor_type_id == TypeId::kNumberTypeInt64) {
    const int64_t *elem_ptr = static_cast<const int64_t *>(index_cpu->data_c());
    for (size_t i = 0; i < elem_num; i++) {
      index_vector.push_back(elem_ptr[i]);
    }
  } else if (tensor_type_id == TypeId::kNumberTypeInt32) {
    const int32_t *elem_ptr = static_cast<const int32_t *>(index_cpu->data_c());
    for (size_t i = 0; i < elem_num; i++) {
      index_vector.push_back(elem_ptr[i]);
    }
  } else if (tensor_type_id == TypeId::kNumberTypeInt16) {
    const int16_t *elem_ptr = static_cast<const int16_t *>(index_cpu->data_c());
    for (size_t i = 0; i < elem_num; i++) {
      index_vector.push_back(elem_ptr[i]);
    }
  } else if (tensor_type_id == TypeId::kNumberTypeInt8) {
    const int8_t *elem_ptr = static_cast<const int8_t *>(index_cpu->data_c());
    for (size_t i = 0; i < elem_num; i++) {
      index_vector.push_back(elem_ptr[i]);
    }
  } else if (tensor_type_id == TypeId::kNumberTypeUInt8) {
    const uint8_t *elem_ptr = static_cast<const uint8_t *>(index_cpu->data_c());
    for (size_t i = 0; i < elem_num; i++) {
      index_vector.push_back(elem_ptr[i]);
    }
  } else {
    MS_LOG(EXCEPTION) << "Unsupported data type for index conversion";
  }

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input, dim_imm, index_vector, value]() {
    MS_LOG(DEBUG) << "Run device task "
                  << " end";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    if (index_vector.empty() && input->DataSize() == 0) {
      MS_LOG(DEBUG) << "aclnnIndexFillTensor throws 'CopyNpuToNpuOp' error when 'self' and 'index' are empty tensors, "
                    << "because we allocated NPU memory for empty 'self' tensor. So we skip launching in this case.";
    } else {
      MS_LOG(DEBUG) << "Run IndexFillScalar device task start";
      LAUNCH_ACLNN(aclnnIndexFillTensor, device_context, op->stream_id(), input, dim_imm, index_vector, value,
                   outputs[0]);
      MS_LOG(DEBUG) << "Run IndexFillScalar device task end";
    }
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
