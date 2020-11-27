/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef DATASET_ENGINE_DATASETOPS_MAP_OP_MAP_JOB_H_
#define DATASET_ENGINE_DATASETOPS_MAP_OP_MAP_JOB_H_

#include <memory>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_row.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class MapJob {
 public:
  // Constructor
  explicit MapJob(std::vector<std::shared_ptr<TensorOp>> operations) : ops_(operations) {}

  // Constructor
  MapJob() = default;

  // Destructor
  ~MapJob() = default;

  Status AddOperation(std::shared_ptr<TensorOp> operation) {
    ops_.push_back(operation);
    return Status::OK();
  }

  // A pure virtual run function to execute a particular map job
  virtual Status Run(std::vector<TensorRow> in, std::vector<TensorRow> *out) = 0;

 protected:
  std::vector<std::shared_ptr<TensorOp>> ops_;
};

}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_DATASETOPS_MAP_OP_MAP_JOB_H_
