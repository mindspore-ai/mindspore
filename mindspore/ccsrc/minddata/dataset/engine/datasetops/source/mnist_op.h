/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_MNIST_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_MNIST_OP_H_

#include <memory>
#include <string>
#include <algorithm>
#include <map>
#include <vector>
#include <utility>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/data_buffer.h"
#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/wait_post.h"

namespace mindspore {
namespace dataset {
// Forward declares
template <typename T>
class Queue;

using MnistLabelPair = std::pair<std::shared_ptr<Tensor>, uint32_t>;

class MnistOp : public ParallelOp, public RandomAccessOp {
 public:
  class Builder {
   public:
    // Constructor for Builder class of MnistOp
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
    // @param int32_t op_connector_size
    // @return Builder setter method returns reference to the builder.
    Builder &SetOpConnectorSize(int32_t op_connector_size) {
      builder_op_connector_size_ = op_connector_size;
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
    // @return
    Builder &SetDir(const std::string &dir) {
      builder_dir_ = dir;
      return *this;
    }

    // Setter method
    // @param const std::string &usage
    // @return
    Builder &SetUsage(const std::string &usage) {
      builder_usage_ = usage;
      return *this;
    }
    // Check validity of input args
    // @return Status The status code returned
    Status SanityCheck();

    // The builder "Build" method creates the final object.
    // @param std::shared_ptr<MnistOp> *op - DatasetOp
    // @return Status The status code returned
    Status Build(std::shared_ptr<MnistOp> *op);

   private:
    std::string builder_dir_;
    std::string builder_usage_;
    int32_t builder_num_workers_;
    int32_t builder_rows_per_buffer_;
    int32_t builder_op_connector_size_;
    std::shared_ptr<SamplerRT> builder_sampler_;
    std::unique_ptr<DataSchema> builder_schema_;
  };

  // Constructor
  // @param const std::string &usage - Usage of this dataset, can be 'train', 'test' or 'all'
  // @param int32_t num_workers - number of workers reading images in parallel
  // @param int32_t rows_per_buffer - number of images (rows) in each buffer
  // @param std::string folder_path - dir directory of mnist
  // @param int32_t queue_size - connector queue size
  // @param std::unique_ptr<DataSchema> data_schema - the schema of the mnist dataset
  // @param td::unique_ptr<Sampler> sampler - sampler tells MnistOp what to read
  MnistOp(const std::string &usage, int32_t num_workers, int32_t rows_per_buffer, std::string folder_path,
          int32_t queue_size, std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler);

  // Destructor.
  ~MnistOp() = default;

  // Worker thread pulls a number of IOBlock from IOBlock Queue, make a buffer and push it to Connector
  // @param int32_t worker_id - id of each worker
  // @return Status The status code returned
  Status WorkerEntry(int32_t worker_id) override;

  // Main Loop of MnistOp
  // Master thread: Fill IOBlockQueue, then goes to sleep
  // Worker thread: pulls IOBlock from IOBlockQueue, work on it then put buffer to mOutConnector
  // @return Status The status code returned
  Status operator()() override;

  // Method derived from RandomAccess Op, enable Sampler to get all ids for each class
  // @param (std::map<uint64_t, std::vector<uint64_t >> * map - key label, val all ids for this class
  // @return Status The status code returned
  Status GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const override;

  // A print method typically used for debugging
  // @param out
  // @param show_all
  void Print(std::ostream &out, bool show_all) const override;

  // Function to count the number of samples in the MNIST dataset
  // @param dir path to the MNIST directory
  // @param count output arg that will hold the minimum of the actual dataset size and numSamples
  // @return
  static Status CountTotalRows(const std::string &dir, const std::string &usage, int64_t *count);

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return "MnistOp"; }

 private:
  // Initialize Sampler, calls sampler->Init() within
  // @return Status The status code returned
  Status InitSampler();

  // Load a tensor row according to a pair
  // @param row_id_type row_id - id for this tensor row
  // @param ImageLabelPair pair - <imagefile,label>
  // @param TensorRow row - image & label read into this tensor row
  // @return Status The status code returned
  Status LoadTensorRow(row_id_type row_id, const MnistLabelPair &mnist_pair, TensorRow *row);

  // @param const std::vector<int64_t> &keys - keys in ioblock
  // @param std::unique_ptr<DataBuffer> db
  // @return Status The status code returned
  Status LoadBuffer(const std::vector<int64_t> &keys, std::unique_ptr<DataBuffer> *db);

  // Iterate through all members in sampleIds and fill them into IOBlock.
  // @param std::shared_ptr<Tensor> sample_ids -
  // @param std::vector<int64_t> *keys - keys in ioblock
  // @return Status The status code returned
  Status TraversalSampleIds(const std::shared_ptr<Tensor> &sample_ids, std::vector<int64_t> *keys);

  // Check image file stream.
  // @param const std::string *file_name - image file name
  // @param std::ifstream *image_reader - image file stream
  // @param uint32_t num_images - returns the number of images
  // @return Status The status code returned
  Status CheckImage(const std::string &file_name, std::ifstream *image_reader, uint32_t *num_images);

  // Check label stream.
  // @param const std::string &file_name - label file name
  // @param std::ifstream *label_reader - label file stream
  // @param uint32_t num_labels - returns the number of labels
  // @return Status The status code returned
  Status CheckLabel(const std::string &file_name, std::ifstream *label_reader, uint32_t *num_labels);

  // Read 4 bytes of data from a file stream.
  // @param std::ifstream *reader - file stream to read
  // @return uint32_t - read out data
  Status ReadFromReader(std::ifstream *reader, uint32_t *result);

  // Swap endian
  // @param uint32_t val -
  // @return uint32_t - swap endian data
  uint32_t SwapEndian(uint32_t val) const;

  // Read the specified number of images and labels from the file stream
  // @param std::ifstream *image_reader - image file stream
  // @param std::ifstream *label_reader - label file stream
  // @param int64_t read_num - number of image to read
  // @return Status The status code returned
  Status ReadImageAndLabel(std::ifstream *image_reader, std::ifstream *label_reader, size_t index);

  // Parse all mnist dataset files
  // @return Status The status code returned
  Status ParseMnistData();

  // Read all files in the directory
  // @return Status The status code returned
  Status WalkAllFiles();

  // Called first when function is called
  // @return Status The status code returned
  Status LaunchThreadsAndInitOp();

  // reset Op
  // @return Status The status code returned
  Status Reset() override;

  // Private function for computing the assignment of the column name map.
  // @return - Status
  Status ComputeColMap() override;

  int64_t buf_cnt_;
  int64_t row_cnt_;
  std::string folder_path_;  // directory of image folder
  int32_t rows_per_buffer_;
  const std::string usage_;  // can only be either "train" or "test"
  std::unique_ptr<DataSchema> data_schema_;
  std::vector<MnistLabelPair> image_label_pairs_;
  std::vector<std::string> image_names_;
  std::vector<std::string> label_names_;
  std::vector<std::string> image_path_;
  std::vector<std::string> label_path_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_MNIST_OP_H_
