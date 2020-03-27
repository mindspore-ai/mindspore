/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef DATASET_ENGINE_DATASETOPS_SOURCE_VOC_OP_H_
#define DATASET_ENGINE_DATASETOPS_SOURCE_VOC_OP_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dataset/core/tensor.h"
#include "dataset/engine/data_buffer.h"
#include "dataset/engine/data_schema.h"
#include "dataset/engine/datasetops/parallel_op.h"
#include "dataset/engine/datasetops/source/io_block.h"
#include "dataset/engine/datasetops/source/sampler/sampler.h"
#include "dataset/kernels/image/image_utils.h"
#include "dataset/util/path.h"
#include "dataset/util/queue.h"
#include "dataset/util/status.h"
#include "dataset/util/wait_post.h"

namespace mindspore {
namespace dataset {
// Forward declares
template <typename T>
class Queue;

class VOCOp : public ParallelOp, public RandomAccessOp {
 public:
  class Builder {
   public:
    // Constructor for Builder class of ImageFolderOp
    // @param  uint32_t numWrks - number of parallel workers
    // @param dir - directory folder got ImageNetFolder
    Builder();

    // Destructor.
    ~Builder() = default;

    // Setter method.
    // @param const std::string & build_dir
    // @return Builder setter method returns reference to the builder.
    Builder &SetDir(const std::string &build_dir) {
      builder_dir_ = build_dir;
      return *this;
    }

    // Setter method.
    // @param int32_t num_workers
    // @return Builder setter method returns reference to the builder.
    Builder &SetNumWorkers(int32_t num_workers) {
      builder_num_workers_ = num_workers;
      return *this;
    }

    // Setter method.
    // @param int32_t op_connector_size
    // @return Builder setter method returns reference to the builder.
    Builder &SetOpConnectorSize(int32_t op_connector_size) {
      builder_op_connector_size_ = op_connector_size;
      return *this;
    }

    // Setter method.
    // @param int32_t rows_per_buffer
    // @return Builder setter method returns reference to the builder.
    Builder &SetRowsPerBuffer(int32_t rows_per_buffer) {
      builder_rows_per_buffer_ = rows_per_buffer;
      return *this;
    }

    // Setter method.
    // @param int64_t num_samples
    // @return Builder setter method returns reference to the builder.
    Builder &SetNumSamples(int64_t num_samples) {
      builder_num_samples_ = num_samples;
      return *this;
    }

    // Setter method.
    // @param std::shared_ptr<Sampler> sampler
    // @return Builder setter method returns reference to the builder.
    Builder &SetSampler(std::shared_ptr<Sampler> sampler) {
      builder_sampler_ = std::move(sampler);
      return *this;
    }

    // Setter method.
    // @param bool do_decode
    // @return Builder setter method returns reference to the builder.
    Builder &SetDecode(bool do_decode) {
      builder_decode_ = do_decode;
      return *this;
    }

    // Check validity of input args
    // @return = The error code return
    Status SanityCheck();

    // The builder "Build" method creates the final object.
    // @param std::shared_ptr<VOCOp> *op - DatasetOp
    // @return - The error code return
    Status Build(std::shared_ptr<VOCOp> *op);

   private:
    bool builder_decode_;
    std::string builder_dir_;
    int32_t builder_num_workers_;
    int32_t builder_op_connector_size_;
    int32_t builder_rows_per_buffer_;
    int64_t builder_num_samples_;
    std::shared_ptr<Sampler> builder_sampler_;
    std::unique_ptr<DataSchema> builder_schema_;
  };

  // Constructor
  // @param int32_t num_workers - number of workers reading images in parallel
  // @param int32_t rows_per_buffer - number of images (rows) in each buffer
  // @param std::string folder_path - dir directory of VOC
  // @param int32_t queue_size - connector queue size
  // @param int64_t num_samples - number of samples to read
  // @param bool decode - whether to decode images
  // @param std::unique_ptr<DataSchema> data_schema - the schema of the VOC dataset
  // @param std::shared_ptr<Sampler> sampler - sampler tells VOCOp what to read
  VOCOp(int32_t num_workers, int32_t rows_per_buffer, const std::string &folder_path, int32_t queue_size,
        int64_t num_samples, bool decode, std::unique_ptr<DataSchema> data_schema, std::shared_ptr<Sampler> sampler);

  // Destructor
  ~VOCOp() = default;

  // Worker thread pulls a number of IOBlock from IOBlock Queue, make a buffer and push it to Connector
  // @param int32_t workerId - id of each worker
  // @return Status - The error code return
  Status WorkerEntry(int32_t worker_id) override;

  // Main Loop of VOCOp
  // Master thread: Fill IOBlockQueue, then goes to sleep
  // Worker thread: pulls IOBlock from IOBlockQueue, work on it the put buffer to mOutConnector
  // @return Status - The error code return
  Status operator()() override;

  // Method derived from RandomAccessOp, enable Sampler to get numRows
  // @param uint64_t num - to return numRows
  // return Status - The error code return
  Status GetNumSamples(int64_t *num) const override;

  // Method derived from RandomAccessOp, enable Sampler to get total number of rows in dataset
  // @param uint64_t num - to return numRows
  Status GetNumRowsInDataset(int64_t *num) const override;

  // A print method typically used for debugging
  // @param out
  // @param show_all
  void Print(std::ostream &out, bool show_all) const override;

 private:
  // Initialize Sampler, calls sampler->Init() within
  // @return Status - The error code return
  Status InitSampler();

  // Load a tensor row according to image id
  // @param std::string image_id - image id
  // @param TensorRow row - image & target read into this tensor row
  // @return Status - The error code return
  Status LoadTensorRow(const std::string &image_id, TensorRow *row);

  // @param const std::string &path - path to the image file
  // @param const ColDescriptor &col - contains tensor implementation and datatype
  // @param std::shared_ptr<Tensor> tensor - return
  // @return Status - The error code return
  Status ReadImageToTensor(const std::string &path, const ColDescriptor &col, std::shared_ptr<Tensor> *tensor);

  // @param const std::vector<uint64_t> &keys - keys in ioblock
  // @param std::unique_ptr<DataBuffer> db
  // @return Status - The error code return
  Status LoadBuffer(const std::vector<int64_t> &keys, std::unique_ptr<DataBuffer> *db);

  Status ParseImageIds();

  Status TraverseSampleIds(const std::shared_ptr<Tensor> &sample_ids, std::vector<int64_t> *keys);

  // Called first when function is called
  // @return Status - The error code return
  Status LaunchThreadsAndInitOp();

  Status Reset() override;

  bool decode_;
  uint64_t row_cnt_;
  uint64_t buf_cnt_;
  int64_t num_rows_;
  int64_t num_samples_;
  std::string folder_path_;
  int32_t rows_per_buffer_;
  std::shared_ptr<Sampler> sampler_;
  std::unique_ptr<DataSchema> data_schema_;

  WaitPost wp_;
  std::vector<std::string> image_ids_;
  std::unordered_map<std::string, int32_t> col_name_map_;
  QueueList<std::unique_ptr<IOBlock>> io_block_queues_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_ENGINE_DATASETOPS_SOURCE_VOC_OP_H_
