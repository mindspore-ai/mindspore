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
#include <set>
#include <string>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_row.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
namespace util {
static inline Status RebuildMapErrorMsg(const TensorRow &input_row, const std::string &op_name, Status *rc) {
  std::string err_msg = "";
  // Need to remove the suffix "Op" which length is 2
  std::string abbr_op_name = op_name.substr(0, op_name.length() - 2);
  err_msg += "map operation: [" + abbr_op_name + "] failed. ";
  if (input_row.getPath().size() > 0 && !input_row.getPath()[0].empty()) {
    err_msg += "The corresponding data file is: " + input_row.getPath()[0];
    if (input_row.getPath().size() > 1) {
      std::set<std::string> path_set;
      (void)path_set.insert(input_row.getPath()[0]);
      for (size_t j = 1; j < input_row.getPath().size(); j++) {
        if (!input_row.getPath()[j].empty() && path_set.find(input_row.getPath()[j]) == path_set.end()) {
          err_msg += ", " + input_row.getPath()[j];
          (void)path_set.insert(input_row.getPath()[j]);
        }
      }
    }
    err_msg += ". ";
  }
  std::string tensor_err_msg = rc->GetErrDescription();
  if (rc->GetLineOfCode() < 0) {
    err_msg += "Error description:\n";
  }
  err_msg += tensor_err_msg;
  if (abbr_op_name == "PyFunc") {
    RETURN_STATUS_ERROR(StatusCode::kMDPyFuncException, err_msg);
  }
  (void)rc->SetErrDescription(err_msg);
  return *rc;
}
}  // namespace util
class MapJob {
 public:
  // Constructor
  explicit MapJob(std::vector<std::shared_ptr<TensorOp>> operations) : ops_(operations) {}

  // Constructor
  MapJob() = default;

  // Destructor
  virtual ~MapJob() = default;

  Status AddOperation(std::shared_ptr<TensorOp> operation) {
    ops_.push_back(operation);
    return Status::OK();
  }

  // A pure virtual run function to execute a particular map job
  virtual Status Run(std::vector<TensorRow> in, std::vector<TensorRow> *out) = 0;

#if !defined(BUILD_LITE) && defined(ENABLE_D)
  // A pure virtual run function to execute a particular map job for Ascend910B DVPP
  virtual Status Run(std::vector<TensorRow> in, std::vector<TensorRow> *out,
                     mindspore::device::DeviceContext *device_context, const size_t &stream_id) = 0;
#endif

  virtual MapTargetDevice Type() = 0;

 protected:
  std::vector<std::shared_ptr<TensorOp>> ops_;
};

}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_DATASETOPS_MAP_OP_MAP_JOB_H_
