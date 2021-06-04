/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_CLUE_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_CLUE_OP_H_

#include <memory>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

#include "minddata/dataset/util/auto_index.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/engine/datasetops/source/nonmappable_leaf_op.h"
#include "minddata/dataset/engine/jagged_connector.h"

namespace mindspore {
namespace dataset {
using StringIndex = AutoIndexObj<std::string>;
using ColKeyMap = std::map<std::string, std::vector<std::string>>;

class JaggedConnector;

class ClueOp : public NonMappableLeafOp {
 public:
  // Constructor of ClueOp
  ClueOp(int32_t num_workers, int64_t num_samples, int32_t worker_connector_size, ColKeyMap cols_to_keyword,
         std::vector<std::string> clue_files_list, int32_t op_connector_size, bool shuffle_files, int32_t num_devices,
         int32_t device_id);

  // Default destructor
  ~ClueOp() = default;

  // A print method typically used for debugging
  // @param out - The output stream to write output to
  // @param show_all - A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override;

  // Instantiates the internal queues and connectors
  // @return Status - the error code returned
  Status Init() override;

  // Get total rows in files.
  // @param files - all clue files.
  // @param count - number of rows.
  // @return Status - the error coed returned.
  static Status CountAllFileRows(const std::vector<std::string> &files, int64_t *count);

  // File names getter
  // @return Vector of the input file names
  std::vector<std::string> FileNames() { return clue_files_list_; }

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return "ClueOp"; }

 private:
  // Reads a clue file and loads the data into multiple TensorRows.
  // @param file - the file to read.
  // @param start_offset - the start offset of file.
  // @param end_offset - the end offset of file.
  // @param worker_id - the id of the worker that is executing this function.
  // @return Status - the error code returned.
  Status LoadFile(const std::string &file, int64_t start_offset, int64_t end_offset, int32_t worker_id) override;

  // Fill the IOBlockQueue.
  // @para i_keys - keys of file to fill to the IOBlockQueue
  // @return Status - the error code returned.
  Status FillIOBlockQueue(const std::vector<int64_t> &i_keys) override;

  // Calculate number of rows in each shard.
  // @return Status - the error code returned.
  Status CalculateNumRowsPerShard() override;

  // Count number of rows in each file.
  // @param filename - clue file name.
  // @return int64_t - the total number of rows in file.
  int64_t CountTotalRows(const std::string &file);

  // @return Status - the error code returned.
  Status GetValue(const nlohmann::json &js, std::vector<std::string> key_chain, std::shared_ptr<Tensor> *t);

  // Private function for computing the assignment of the column name map.
  // @return - Status
  Status ComputeColMap() override;

  std::vector<std::string> clue_files_list_;
  ColKeyMap cols_to_keyword_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_CLUE_OP_H_
