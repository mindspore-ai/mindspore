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
class CelebAOp : public ParallelOp, RandomAccessOp {
 public:
  class Builder {
   public:
    // Constructor for Builder class of CelebAOp
    // @return Builder setter method returns reference to the builder.
    Builder();

    // Destructor.
    ~Builder() = default;

    // Setter method
    // @param int32_t rows_per_buffer
    // @return Builder setter method returns reference to the builder.
    Builder &SetRowsPerBuffer(int32_t rows_per_buffer) {
      builder_rows_per_buffer_ = rows_per_buffer;
      return *this;
    }

    // Setter method
    // @param int32_t size
    // @return Builder setter method returns reference to the builder.
    Builder &SetOpConnectorSize(int32_t size) {
      builder_op_connector_size_ = size;
      return *this;
    }

    // Setter method
    // @param std::set<std::string> & exts, file extensions to be read
    // @return Builder setter method returns reference to the builder.
    Builder &SetExtensions(const std::set<std::string> &exts) {
      builder_extensions_ = exts;
      return *this;
    }

    // Setter method
    // @param bool decode
    // @return Builder setter method returns reference to the builder.
    Builder &SetDecode(bool decode) {
      builder_decode_ = decode;
      return *this;
    }

    // Setter method
    // @param int32_t num_workers
    // @return Builder setter method returns reference to the builder.
    Builder &SetNumWorkers(int32_t num_workers) {
      builder_num_workers_ = num_workers;
      return *this;
    }

    // Setter method
    // @param std::shared_ptr<Sampler> sampler
    // @return Builder setter method returns reference to the builder.
    Builder &SetSampler(std::shared_ptr<SamplerRT> sampler) {
      builder_sampler_ = std::move(sampler);
      return *this;
    }

    // Setter method
    // @param const std::string &dir
    // @return Builder setter method returns reference to the builder.
    Builder &SetCelebADir(const std::string &dir) {
      builder_dir_ = dir;
      return *this;
    }

    // Setter method
    // @param const std::string usage: type to be read
    // @return Builder setter method returns reference to the builder.
    Builder &SetUsage(const std::string &usage) {
      builder_usage_ = usage;
      return *this;
    }
    // Check validity of input args
    // @return Status The status code returned
    Status SanityCheck();

    // The builder "build" method creates the final object.
    // @param std::shared_ptr<CelebAOp> *op - DatasetOp
    // @return Status The status code returned
    Status Build(std::shared_ptr<CelebAOp> *op);

   private:
    bool builder_decode_;
    std::string builder_dir_;
    int32_t builder_num_workers_;
    int32_t builder_rows_per_buffer_;
    int32_t builder_op_connector_size_;
    std::set<std::string> builder_extensions_;
    std::shared_ptr<SamplerRT> builder_sampler_;
    std::unique_ptr<DataSchema> builder_schema_;
    std::string builder_usage_;
  };

  // Constructor
  // @param int32_t - num_workers - Num of workers reading images in parallel
  // @param int32_t - rows_per_buffer Number of images (rows) in each buffer
  // @param std::string - dir directory of celeba dataset
  // @param int32_t queueSize - connector queue size
  // @param std::unique_ptr<Sampler> sampler - sampler tells CelebAOp what to read
  CelebAOp(int32_t num_workers, int32_t rows_per_buffer, const std::string &dir, int32_t queue_size, bool decode,
           const std::string &usage, const std::set<std::string> &exts, std::unique_ptr<DataSchema> schema,
           std::shared_ptr<SamplerRT> sampler);

  ~CelebAOp() override = default;

  // Main Loop of CelebAOp
  // Master thread: Fill IOBlockQueue, then goes to sleep
  // Worker thread: pulls IOBlock from IOBlockQueue, work on it then put buffer to mOutConnector
  // @return Status The status code returned
  Status operator()() override;

  // Worker thread pulls a number of IOBlock from IOBlock Queue, make a buffer and push it to Connector
  // @param int32_t worker_id - id of each worker
  // @return Status The status code returned
  Status WorkerEntry(int32_t worker_id) override;

  // A print method typically used for debugging
  // @param out
  // @param show_all
  void Print(std::ostream &out, bool show_all) const override;

  // Method in operator(), to fill IOBlockQueue
  // @param std::unique_ptr<DataBuffer> sampler_buffer - to fill IOBlockQueue
  // @return Status The status code returned
  Status AddIOBlock(std::unique_ptr<DataBuffer> *data_buffer);

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return "CelebAOp"; }

 private:
  // Called first when function is called
  // @return
  Status LaunchThreadsAndInitOp();

  // Parse attribute file
  // @return
  Status ParseAttrFile();

  // Parse each image line in attribute file
  // @return
  Status ParseImageAttrInfo();

  // Split attribute info with space
  // @param std::string - line - Line from att or partition file
  // @return std::vector<std::string> - string after split
  std::vector<std::string> Split(const std::string &line);

  // @param const std::vector<int64_t> &keys - keys in ioblock
  // @param std::unique_ptr<DataBuffer> db
  // @return Status The status code returned
  Status LoadBuffer(const std::vector<int64_t> &keys, std::unique_ptr<DataBuffer> *db);

  // Load a tensor row according to a pair
  // @param row_id_type row_id - id for this tensor row
  // @param std::pair - <image_file,<label>>
  // @param TensorRow row - image & label read into this tensor row
  // @return Status The status code returned
  Status LoadTensorRow(row_id_type row_id, const std::pair<std::string, std::vector<int32_t>> &image_label,
                       TensorRow *row);

  // Check if need read according to dataset type
  // @return bool - if need read
  bool CheckDatasetTypeValid();

  // reset Op
  // @return Status The status code returned
  Status Reset() override;

  // Private function for computing the assignment of the column name map.
  // @return - Status
  Status ComputeColMap() override;

  int32_t rows_per_buffer_;
  std::string folder_path_;  // directory of celeba folder
  bool decode_;
  std::set<std::string> extensions_;  // extensions allowed
  std::unique_ptr<DataSchema> data_schema_;
  std::unique_ptr<Queue<std::vector<std::string>>> attr_info_queue_;
  int64_t num_rows_in_attr_file_;  // rows number specified in attr file
  std::vector<std::pair<std::string, std::vector<int32_t>>> image_labels_vec_;
  std::string usage_;
  std::ifstream partition_file_;
  std::string attr_file_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_CELEBA_OP_H
