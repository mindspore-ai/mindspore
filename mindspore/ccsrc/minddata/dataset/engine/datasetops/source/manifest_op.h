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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_MANIFEST_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_MANIFEST_OP_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/data_buffer.h"
#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/util/services.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/wait_post.h"

namespace mindspore {
namespace dataset {
class ManifestOp : public ParallelOp, public RandomAccessOp {
 public:
  class Builder {
   public:
    // Constructor for Builder class of ManifestOp
    Builder();

    // Destructor
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
    // @param const std::map<std::string, int32_t>& map - a class name to label map
    // @return
    Builder &SetClassIndex(const std::map<std::string, int32_t> &map) {
      builder_labels_to_read_ = map;
      return *this;
    }

    // Setter method
    // @param bool do_decode
    // @return Builder setter method returns reference to the builder.
    Builder &SetDecode(bool do_decode) {
      builder_decode_ = do_decode;
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
    // @param const std::string & dir
    // @return Builder setter method returns reference to the builder.
    Builder &SetManifestFile(const std::string &file) {
      builder_file_ = file;
      return *this;
    }

    // Setter method
    // @param const std::string & dir
    // @return Builder setter method returns reference to the builder.
    Builder &SetUsage(const std::string &usage) {
      builder_usage_ = usage;
      return *this;
    }

    // Check validity of input args
    // @return Status The status code returned
    Status SanityCheck();

    // The builder "build" method creates the final object.
    // @param std::shared_ptr<ManifestOp> *op - DatasetOp
    // @return Status The status code returned
    Status Build(std::shared_ptr<ManifestOp> *op);

   private:
    std::shared_ptr<SamplerRT> builder_sampler_;
    bool builder_decode_;

    std::string builder_file_;
    int32_t builder_num_workers_;
    int32_t builder_rows_per_buffer_;
    int32_t builder_op_connector_size_;
    std::unique_ptr<DataSchema> builder_schema_;
    std::string builder_usage_;
    std::map<std::string, int32_t> builder_labels_to_read_;
  };

  // Constructor
  // @param int32_t num_works - Num of workers reading images in parallel
  // @param int32_t - rows_per_buffer Number of images (rows) in each buffer
  // @param std::string - file list of Manifest
  // @param int32_t queue_size - connector queue size
  // @param td::unique_ptr<Sampler> sampler - sampler tells ImageFolderOp what to read
  ManifestOp(int32_t num_works, int32_t rows_per_buffer, std::string file, int32_t queue_size, bool decode,
             const std::map<std::string, int32_t> &class_index, std::unique_ptr<DataSchema> data_schema,
             std::shared_ptr<SamplerRT> sampler, std::string usage);
  // Destructor.
  ~ManifestOp() = default;

  // Worker thread pulls a number of IOBlock from IOBlock Queue, make a buffer and push it to Connector
  // @param int32_t worker_id - id of each worker
  // @return Status The status code returned
  Status WorkerEntry(int32_t worker_id) override;

  // Main Loop of ManifestOp
  // Master thread: Fill IOBlockQueue, then goes to sleep
  // Worker thread: pulls IOBlock from IOBlockQueue, work on it then put buffer to mOutConnector
  // @return Status The status code returned
  Status operator()() override;

  // Method derived from RandomAccess Op, enable Sampler to get all ids for each class
  // @param (std::map<int64_t, std::vector<int64_t >> * map - key label, val all ids for this class
  // @return Status The status code returned
  Status GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const override;

  // A print method typically used for debugging
  // @param out
  // @param show_all
  void Print(std::ostream &out, bool show_all) const override;

  /// \brief Counts the total number of rows in Manifest
  /// \param[in] file Dataset file path
  /// \param[in] input_class_indexing Input map of class index
  /// \param[in] usage Dataset usage
  /// \param[out] count Number of rows counted
  /// \param[out] numClasses Number of classes counted
  /// \return Status of the function
  static Status CountTotalRows(const std::string &file, const std::map<std::string, int32_t> &map,
                               const std::string &usage, int64_t *count, int64_t *numClasses);

#ifdef ENABLE_PYTHON
  // Get str-to-int mapping from label name to index
  static Status GetClassIndexing(const std::string &file, const py::dict &dict, const std::string &usage,
                                 std::map<std::string, int32_t> *output_class_indexing);
#endif

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return "ManifestOp"; }

  /// \brief Base-class override for GetNumClasses
  /// \param[out] num_classes the number of classes
  /// \return Status of the function
  Status GetNumClasses(int64_t *num_classes) override;

  /// \brief Gets the class indexing
  /// \return Status - The status code return
  Status GetClassIndexing(std::vector<std::pair<std::string, std::vector<int32_t>>> *output_class_indexing) override;

 private:
  // Initialize Sampler, calls sampler->Init() within
  // @return Status The status code returned
  Status InitSampler();

  // Method in operator(), to fill IOBlockQueue
  // @param std::unique_ptr<DataBuffer> sampler_buffer - to fill IOBlockQueue
  // @return Status The status code returned
  Status AddIoBlock(std::unique_ptr<DataBuffer> *sampler_buffer);

  // Load a tensor row according to a pair
  // @param row_id_type row_id - id for this tensor row
  // @param std::pair<std::string, std::vector<std::string>> - <imagefile, <label1, label2...>>
  // @param TensorRow row - image & label read into this tensor row
  // @return Status The status code returned
  Status LoadTensorRow(row_id_type row_id, const std::pair<std::string, std::vector<std::string>> &data,
                       TensorRow *row);

  // @param const std::vector<int64_t> &keys - keys in ioblock
  // @param std::unique_ptr<DataBuffer> db
  // @return Status The status code returned
  Status LoadBuffer(const std::vector<int64_t> &keys, std::unique_ptr<DataBuffer> *db);

  // Parse manifest file to get image path and label and so on.
  // @return Status The status code returned
  Status ParseManifestFile();

  // Called first when function is called
  // @return Status The status code returned
  Status LaunchThreadsAndInitOp();

  // reset Op
  // @return Status The status code returned
  Status Reset() override;

  // Check if image ia valid.Only support JPEG/PNG/GIF/BMP
  // @return
  Status CheckImageType(const std::string &file_name, bool *valid);

  // Count label index,num rows and num samples
  // @return Status The status code returned
  Status CountDatasetInfo();

  // Private function for computing the assignment of the column name map.
  // @return - Status
  Status ComputeColMap() override;

  int32_t rows_per_buffer_;
  int64_t io_block_pushed_;
  int64_t row_cnt_;
  int64_t sampler_ind_;
  std::unique_ptr<DataSchema> data_schema_;
  std::string file_;  // file that store the information of images
  std::map<std::string, int32_t> class_index_;
  bool decode_;
  std::string usage_;
  int64_t buf_cnt_;

  std::map<std::string, int32_t> label_index_;
  std::vector<std::pair<std::string, std::vector<std::string>>> image_labelname_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_MANIFEST_OP_H_
