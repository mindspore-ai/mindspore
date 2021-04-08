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

#include <memory>
#include <vector>
#include <utility>
#include <set>
#include <string>
#include "minddata/dataset/engine/datasetops/map_op/cpu_map_job.h"

namespace mindspore {
namespace dataset {

// Constructor
CpuMapJob::CpuMapJob() = default;

// Constructor
CpuMapJob::CpuMapJob(std::vector<std::shared_ptr<TensorOp>> operations) : MapJob(std::move(operations)) {}

// Destructor
CpuMapJob::~CpuMapJob() = default;

// A function to execute a cpu map job
Status CpuMapJob::Run(std::vector<TensorRow> in, std::vector<TensorRow> *out) {
  int32_t num_rows = in.size();
  for (int32_t row = 0; row < num_rows; row++) {
    TensorRow input_row = in[row];
    TensorRow result_row;
    for (size_t i = 0; i < ops_.size(); i++) {
      // Call compute function for cpu
      Status rc = ops_[i]->Compute(input_row, &result_row);
      if (rc.IsError()) {
        std::string err_msg = "";
        std::string op_name = ops_[i]->Name();
        std::string abbr_op_name = op_name.substr(0, op_name.length() - 2);
        err_msg += "map operation: [" + abbr_op_name + "] failed. ";
        if (input_row.getPath().size() > 0 && !input_row.getPath()[0].empty()) {
          err_msg += "The corresponding data files: " + input_row.getPath()[0];
          if (input_row.getPath().size() > 1) {
            std::set<std::string> path_set;
            path_set.insert(input_row.getPath()[0]);
            for (auto j = 1; j < input_row.getPath().size(); j++) {
              if (!input_row.getPath()[j].empty() && path_set.find(input_row.getPath()[j]) == path_set.end()) {
                err_msg += ", " + input_row.getPath()[j];
                path_set.insert(input_row.getPath()[j]);
              }
            }
          }
          err_msg += ". ";
        }
        std::string tensor_err_msg = rc.GetErrDescription();
        if (rc.GetLineOfCode() < 0) {
          err_msg += "Error description:\n";
        }
        err_msg += tensor_err_msg;
        rc.SetErrDescription(err_msg);
        RETURN_IF_NOT_OK(rc);
      }

      // Assign result_row to to_process for the next TensorOp processing, except for the last TensorOp in the list.
      if (i + 1 < ops_.size()) {
        input_row = std::move(result_row);
      }
    }
    out->push_back(std::move(result_row));
  }
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
