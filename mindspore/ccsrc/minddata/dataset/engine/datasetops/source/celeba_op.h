/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_CELEBA_OP_H
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_CELEBA_OP_H

#include <string>
#include <set>
#include <memory>
#include <vector>
#include <utility>
#include <fstream>

#include "minddata/dataset/util/status.h"
#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/engine/datasetops/source/mappable_leaf_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/engine/datasetops/source/io_block.h"

#define CLOSE_FILE(attr_file, pairition_file) \
  do {                                        \
    attr_file.close();                        \
    if (pairition_file.is_open()) {           \
      pairition_file.close();                 \
    }                                         \
  } while (false)

namespace mindspore {
namespace dataset {
class CelebAOp : public MappableLeafOp {
 public:
  // Constructor
  // @param int32_t - num_workers - Num of workers reading images in parallel
  // @param std::string - dir directory of celeba dataset
  // @param int32_t queueSize - connector queue size
  // @param bool decode - decode the images after reading
  // @param std::string usage - specify the train, valid, test part or all parts of dataset
  // @param std::set<std::string> exts - list of file extensions to be included in the dataset
  // @param std::unique_ptr<DataSchema> schema - path to the JSON schema file or schema object
  // @param std::unique_ptr<Sampler> sampler - sampler tells CelebAOp what to read
  CelebAOp(int32_t num_workers, const std::string &dir, int32_t queue_size, bool decode, const std::string &usage,
           const std::set<std::string> &exts, std::unique_ptr<DataSchema> schema, std::shared_ptr<SamplerRT> sampler);

  ~CelebAOp() override = default;

  // A print method typically used for debugging
  // @param out
  // @param show_all
  void Print(std::ostream &out, bool show_all) const override;

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return "CelebAOp"; }

 private:
  // Called first when function is called
  // @return
  Status LaunchThreadsAndInitOp() override;

  /// Parse attribute file
  /// @return
  Status ParseAttrFile();

  /// Parse each image line in attribute file
  /// @return
  Status ParseImageAttrInfo();

  /// Split attribute info with space
  /// @param std::string - line - Line from att or partition file
  /// @return std::vector<std::string> - string after split
  std::vector<std::string> Split(const std::string &line);

  // Load a tensor row according to a pair
  // @param row_id_type row_id - id for this tensor row
  // @param std::pair - <image_file,<label>>
  // @param TensorRow row - image & label read into this tensor row
  // @return Status The status code returned
  Status LoadTensorRow(row_id_type row_id, TensorRow *row) override;

  /// Check if need read according to dataset type
  /// @return bool - if need read
  bool CheckDatasetTypeValid();

  // Private function for computing the assignment of the column name map.
  // @return - Status
  Status ComputeColMap() override;

  std::string folder_path_;  // directory of celeba folder
  bool decode_;
  std::set<std::string> extensions_;  /// extensions allowed
  std::unique_ptr<DataSchema> data_schema_;
  std::unique_ptr<Queue<std::vector<std::string>>> attr_info_queue_;
  int64_t num_rows_in_attr_file_;  /// rows number specified in attr file
  std::vector<std::pair<std::string, std::vector<int32_t>>> image_labels_vec_;
  std::string usage_;
  std::ifstream partition_file_;
  std::string attr_file_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_CELEBA_OP_H
