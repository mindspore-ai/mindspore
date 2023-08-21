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

#include "minddata/dataset/engine/datasetops/map_op/npu_map_job.h"

#include <set>
#include <string>
#include <utility>

#include "minddata/dataset/core/device_tensor_ascend910b.h"

namespace mindspore {
namespace dataset {

// Constructor
NpuMapJob::NpuMapJob() = default;

// Constructor
NpuMapJob::NpuMapJob(std::vector<std::shared_ptr<TensorOp>> operations) : MapJob(std::move(operations)) {}

// Destructor
NpuMapJob::~NpuMapJob() = default;

// A function to execute a npu map job
Status NpuMapJob::Run(std::vector<TensorRow> in, std::vector<TensorRow> *out, device::DeviceContext *device_context,
                      const size_t &stream_id) {
  RETURN_UNEXPECTED_IF_NULL(out);
  RETURN_UNEXPECTED_IF_NULL(device_context);
  int32_t num_rows = in.size();

  // create the device tensor which copy the data from host to device
  std::vector<std::vector<std::shared_ptr<DeviceTensorAscend910B>>> device_in(num_rows);
  uint32_t i = 0;
  for (auto &tensor_row : in) {
    for (auto &tensor : tensor_row) {
      std::shared_ptr<DeviceTensorAscend910B> device_tensor = nullptr;
      RETURN_IF_NOT_OK(DeviceTensorAscend910B::CreateDeviceTensor(tensor, device_context, stream_id, &device_tensor));
      device_in[i].push_back(std::move(device_tensor));
    }
    i += 1;
  }

  std::vector<std::vector<std::shared_ptr<DeviceTensorAscend910B>>> device_out;

  for (int32_t row = 0; row < num_rows; row++) {
    std::vector<std::shared_ptr<DeviceTensorAscend910B>> input_row = device_in[row];
    std::vector<std::shared_ptr<DeviceTensorAscend910B>> result_row;
    for (size_t i = 0; i < ops_.size(); i++) {
      // Call compute function for npu
      Status rc = ops_[i]->Compute(input_row, &result_row);
      if (rc.IsError()) {
        std::string op_name = ops_[i]->Name();
        RETURN_IF_NOT_OK(util::RebuildMapErrorMsg(in[row], op_name, &rc));
      }

      // release the device memory first
      for (auto &item : input_row) {
        device_context->device_res_manager_->FreeMemory(item->GetDeviceAddress().get());
      }

      // Assign result_row to to_process for the next TensorOp processing, except for the last TensorOp in the list.
      if (i + 1 < ops_.size()) {
        input_row = std::move(result_row);
      }
    }
    device_out.push_back(std::move(result_row));
  }

  // copy the data from device to host
  for (auto &tensor_row : device_out) {
    TensorRow result_row;
    for (auto &tensor : tensor_row) {
      std::shared_ptr<Tensor> host_out;
      CHECK_FAIL_RETURN_UNEXPECTED(tensor->ToHostTensor(&host_out), "Copy tensor from device to host failed.");
      result_row.push_back(std::move(host_out));
    }
    out->push_back(std::move(result_row));
  }

  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
