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
#ifndef DATASET_ENGINE_DATASETOPS_MAP_OP_NPU_MAP_JOB_H_
#define DATASET_ENGINE_DATASETOPS_MAP_OP_NPU_MAP_JOB_H_

#include <memory>
#include <vector>
#include "minddata/dataset/engine/datasetops/map_op/map_job.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"

namespace mindspore {
namespace dataset {
class NpuMapJob : public MapJob {
 public:
  // Constructor
  NpuMapJob();

  // Constructor
  explicit NpuMapJob(std::vector<std::shared_ptr<TensorOp>> operations);

  // Destructor
  ~NpuMapJob();

  // A pure virtual run function to execute a npu map job
  Status Run(const std::vector<TensorRow> &in, std::vector<TensorRow> *out) override {
    RETURN_STATUS_UNEXPECTED("The run operation is not implemneted in NPU platform.");
  }

  // A pure virtual run function to execute a npu map job for Ascend910B DVPP
  Status Run(const std::vector<TensorRow> &in, std::vector<TensorRow> *out,
             mindspore::device::DeviceContext *device_context, const size_t &stream_id) override;

  MapTargetDevice Type() override { return MapTargetDevice::kAscend910B; }
};

}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_DATASETOPS_MAP_OP_NPU_MAP_JOB_H_
