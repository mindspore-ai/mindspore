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

#include "minddata/dataset/core/device_tensor_ascend910b.h"
#include "minddata/dataset/kernels/image/image_utils.h"

namespace mindspore {
namespace dataset {

// Constructor
NpuMapJob::NpuMapJob() = default;

// Constructor
NpuMapJob::NpuMapJob(std::vector<std::shared_ptr<TensorOp>> operations) : MapJob(std::move(operations)) {}

// Destructor
NpuMapJob::~NpuMapJob() = default;

// A function to execute a npu map job
Status NpuMapJob::Run(std::vector<TensorRow> in, std::vector<TensorRow> *out,
                      mindspore::device::DeviceContext *device_context, const size_t &stream_id) {
  RETURN_UNEXPECTED_IF_NULL(out);
  RETURN_UNEXPECTED_IF_NULL(device_context);
  size_t num_rows = in.size();

  // create the device tensor which copy the data from host to device
  std::vector<std::vector<std::shared_ptr<DeviceTensorAscend910B>>> device_in(num_rows);
  uint32_t i = 0;
  for (auto &tensor_row : in) {
    for (auto &tensor : tensor_row) {
      // if the first op is Decode, confirm that it is in jpeg format.
      if (ops_[0]->Name() == kDvppDecodeOp) {
        CHECK_FAIL_RETURN_UNEXPECTED(tensor->shape().Rank() == 1,
                                     "[DvppDecode] Invalid data shape. Currently only support 1D. Its rank is: " +
                                       std::to_string(tensor->shape().Rank()));
        CHECK_FAIL_RETURN_UNEXPECTED(
          IsNonEmptyJPEG(tensor) == true,
          "[DvppDecode] Invalid image type. Currently only support JPG. Its shape is: " + tensor->shape().ToString());
        CHECK_FAIL_RETURN_UNEXPECTED(
          tensor->type() == DataType::DE_UINT8,
          "[DvppDecode] Invalid data type. Currently only support uint8. Its type is: " + tensor->type().ToString());
      }
      std::shared_ptr<DeviceTensorAscend910B> device_tensor = nullptr;
      // here we use the first op's IsHWC() to create device tensor
      if (ops_[0]->Name() == kDvppConvertColorOp) {
        std::vector<int> channels = {1, 3, 4};
        RETURN_IF_NOT_OK(DeviceTensorAscend910B::CreateDeviceTensor(tensor, device_context, stream_id, &device_tensor,
                                                                    ops_[0]->IsHWC(), channels));
      } else {
        RETURN_IF_NOT_OK(DeviceTensorAscend910B::CreateDeviceTensor(tensor, device_context, stream_id, &device_tensor,
                                                                    ops_[0]->IsHWC()));
      }
      device_in[i].push_back(std::move(device_tensor));
    }
    i += 1;
  }

  std::vector<std::vector<std::shared_ptr<DeviceTensorAscend910B>>> device_out;
  std::vector<std::vector<std::vector<std::shared_ptr<DeviceTensorAscend910B>>>> hold_input_lists;

  for (int32_t row = 0; row < num_rows; row++) {
    std::vector<std::shared_ptr<DeviceTensorAscend910B>> input_row = device_in[row];
    std::vector<std::shared_ptr<DeviceTensorAscend910B>> result_row;

    // hold all the inputs and release them when the npu_map_job finish
    std::vector<std::vector<std::shared_ptr<DeviceTensorAscend910B>>> hold_input_list;

    for (size_t i = 0; i < ops_.size(); i++) {
      // if the op is Decode, we should get the height and width form JPEG header and create the output tensor first
      int img_width = 0;
      int img_height = 0;
      if (i == 0 && ops_[i]->Name() == kDvppDecodeOp) {
        for (int32_t k = 0; k < in[0].size(); k++) {
          RETURN_IF_NOT_OK(GetJpegImageInfo(in[0][k], &img_width, &img_height));
          TensorShape shape{1, img_height, img_width, 3};
          DataType type(DataType::DE_UINT8);
          std::shared_ptr<DeviceTensorAscend910B> device_tensor = nullptr;
          RETURN_IF_NOT_OK(DeviceTensorAscend910B::CreateDeviceTensor(shape, type, input_row[k]->GetDeviceContext(),
                                                                      input_row[k]->GetStreamID(), &device_tensor));
          result_row.push_back(std::move(device_tensor));
        }
      }
      // Call compute function for npu
      Status rc = ops_[i]->Compute(input_row, &result_row);
      if (rc.IsError()) {
        std::string op_name = ops_[i]->Name();
        RETURN_IF_NOT_OK(util::RebuildMapErrorMsg(in[row], op_name, &rc));
      }

      // move the input to the hold_input_list first
      hold_input_list.push_back(std::move(input_row));

      // Assign result_row to to_process for the next TensorOp processing, except for the last TensorOp in the list.
      if (i + 1 < ops_.size()) {
        input_row = std::move(result_row);
      }
    }
    device_out.push_back(std::move(result_row));
    hold_input_lists.push_back(std::move(hold_input_list));
  }

  // Because we do ToHostTensor, we should sync first
  if (!device_context->device_res_manager_->SyncStream(stream_id)) {
    std::string err_msg = "SyncStream stream id: " + std::to_string(stream_id) + " failed.";
    RETURN_STATUS_UNEXPECTED(err_msg);
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

  // release all the device memory
  for (auto &hold_input_list : hold_input_lists) {
    for (auto &tensor_list : hold_input_list) {
      for (auto &item : tensor_list) {
        if (!item->ReleaseDeviceMemory()) {
          std::string err_msg = "Release the device memory failed after the dvpp ops executed.";
          RETURN_STATUS_UNEXPECTED(err_msg);
        }
      }
    }
  }

  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
