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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_ALBUM_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_ALBUM_OP_H_

#include <algorithm>
#include <deque>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/data_buffer.h"
#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/util/services.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/wait_post.h"

namespace mindspore {
namespace dataset {
// Forward declares
template <typename T>
class Queue;

// Define row information as a list of file objects to read
using FolderImages = std::shared_ptr<std::pair<std::string, std::queue<std::string>>>;

/// \class AlbumOp album_op.h
class AlbumOp : public ParallelOp, public RandomAccessOp {
 public:
  class Builder {
   public:
    /// \brief Constructor for Builder class of AlbumOp
    Builder();

    /// \brief Destructor.
    ~Builder() = default;

    /// \brief Setter method
    /// \param[in] rows_per_buffer
    /// \return Builder setter method returns reference to the builder
    Builder &SetRowsPerBuffer(int32_t rows_per_buffer) {
      builder_rows_per_buffer_ = rows_per_buffer;
      return *this;
    }

    /// \brief Setter method
    /// \param[in] size
    /// \return Builder setter method returns reference to the builder
    Builder &SetOpConnectorSize(int32_t size) {
      builder_op_connector_size_ = size;
      return *this;
    }

    /// \brief Setter method
    /// \param[in] exts - file extensions to be read
    /// \return Builder setter method returns reference to the builder
    Builder &SetExtensions(const std::set<std::string> &exts) {
      builder_extensions_ = exts;
      return *this;
    }

    /// \brief Setter method
    /// \param[in] do_decode
    /// \return Builder setter method returns reference to the builder
    Builder &SetDecode(bool do_decode) {
      builder_decode_ = do_decode;
      return *this;
    }

    /// \brief Setter method
    /// \param[in] num_workers
    /// \return Builder setter method returns reference to the builder
    Builder &SetNumWorkers(int32_t num_workers) {
      builder_num_workers_ = num_workers;
      return *this;
    }

    /// \brief Setter method
    /// \param[in] sampler
    /// \return Builder setter method returns reference to the builder
    Builder &SetSampler(std::shared_ptr<SamplerRT> sampler) {
      builder_sampler_ = std::move(sampler);
      return *this;
    }

    /// \brief Setter method
    /// \param[in] dir - dataset directory
    /// \return Builder setter method returns reference to the builder
    Builder &SetAlbumDir(const std::string &dir) {
      builder_dir_ = dir;
      return *this;
    }

    /// \brief Setter method
    /// \param[in] file - schema file to load
    /// \return Builder setter method returns reference to the builder
    Builder &SetSchemaFile(const std::string &file) {
      builder_schema_file_ = file;
      return *this;
    }

    /// \brief Setter method
    /// \param[in] columns - input columns
    /// \return Builder setter method returns reference to the builder
    Builder &SetColumnsToLoad(const std::vector<std::string> &columns) {
      builder_columns_to_load_ = columns;
      return *this;
    }

    /// \brief Check validity of input args
    /// \return Status The status code returned
    Status SanityCheck();

    /// \brief The builder "build" method creates the final object.
    /// \param[in, out] std::shared_ptr<AlbumOp> *op - DatasetOp
    /// \return Status The status code returned
    Status Build(std::shared_ptr<AlbumOp> *op);

   private:
    bool builder_decode_;
    std::vector<std::string> builder_columns_to_load_;
    std::string builder_dir_;
    std::string builder_schema_file_;
    int32_t builder_num_workers_;
    int32_t builder_rows_per_buffer_;
    int32_t builder_op_connector_size_;
    std::set<std::string> builder_extensions_;
    std::shared_ptr<SamplerRT> builder_sampler_;
    std::unique_ptr<DataSchema> builder_schema_;
  };

  /// \brief Constructor
  /// \param[in] num_wkrs - Num of workers reading images in parallel
  /// \param[in] rows_per_buffer Number of images (rows) in each buffer
  /// \param[in] file_dir - directory of Album
  /// \param[in] queue_size - connector size
  /// \param[in] do_decode - decode image files
  /// \param[in] exts - set of file extensions to read, if empty, read everything under the dir
  /// \param[in] data_schema - schema of dataset
  /// \param[in] sampler - sampler tells AlbumOp what to read
  AlbumOp(int32_t num_wkrs, int32_t rows_per_buffer, std::string file_dir, int32_t queue_size, bool do_decode,
          const std::set<std::string> &exts, std::unique_ptr<DataSchema> data_schema,
          std::shared_ptr<SamplerRT> sampler);

  /// \brief Destructor.
  ~AlbumOp() = default;

  /// \brief Initialize AlbumOp related var, calls the function to walk all files
  /// \return Status The status code returned
  Status PrescanEntry();

  /// \brief Worker thread pulls a number of IOBlock from IOBlock Queue, make a buffer and push it to Connector
  /// \param[in] int32_t workerId - id of each worker
  /// \return Status The status code returned
  Status WorkerEntry(int32_t worker_id) override;

  /// \brief Main Loop of AlbumOp
  ///     Master thread: Fill IOBlockQueue, then goes to sleep
  ///     Worker thread: pulls IOBlock from IOBlockQueue, work on it then put buffer to mOutConnector
  /// \return Status The status code returned
  Status operator()() override;

  /// \brief A print method typically used for debugging
  /// \param[in] out
  /// \param[in] show_all
  void Print(std::ostream &out, bool show_all) const override;

  /// \brief Check if image ia valid.Only support JPEG/PNG/GIF/BMP
  ///     This function could be optimized to return the tensor to reduce open/closing files
  /// \return bool - if file is bad then return false
  bool CheckImageType(const std::string &file_name, bool *valid);

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return "AlbumOp"; }

 private:
  /// \brief Initialize Sampler, calls sampler->Init() within
  /// \return Status The status code returned
  Status InitSampler();

  /// \brief Load image to tensor row
  /// \param[in] image_file Image name of file
  /// \param[in] col_num Column num in schema
  /// \param[in, out] row Tensor row to push to
  /// \return Status The status code returned
  Status LoadImageTensor(const std::string &image_file, uint32_t col_num, TensorRow *row);

  /// \brief Load vector of ints to tensor, append tensor to tensor row
  /// \param[in] json_obj Json object containing multi-dimensional label
  /// \param[in] col_num Column num in schema
  /// \param[in, out] row Tensor row to push to
  /// \return Status The status code returned
  Status LoadIntArrayTensor(const nlohmann::json &json_obj, uint32_t col_num, TensorRow *row);

  /// \brief Load vector of floatss to tensor, append tensor to tensor row
  /// \param[in] json_obj Json object containing array data
  /// \param[in] col_num Column num in schema
  /// \param[in, out] row Tensor row to push to
  /// \return Status The status code returned
  Status LoadFloatArrayTensor(const nlohmann::json &json_obj, uint32_t col_num, TensorRow *row);

  /// \brief Load string array into a tensor, append tensor to tensor row
  /// \param[in] json_obj Json object containing string tensor
  /// \param[in] col_num Column num in schema
  /// \param[in, out] row Tensor row to push to
  /// \return Status The status code returned
  Status LoadStringArrayTensor(const nlohmann::json &json_obj, uint32_t col_num, TensorRow *row);

  /// \brief Load string into a tensor, append tensor to tensor row
  /// \param[in] json_obj Json object containing string tensor
  /// \param[in] col_num Column num in schema
  /// \param[in, out] row Tensor row to push to
  /// \return Status The status code returned
  Status LoadStringTensor(const nlohmann::json &json_obj, uint32_t col_num, TensorRow *row);

  /// \brief Load float value to tensor row
  /// \param[in] json_obj Json object containing float
  /// \param[in] col_num Column num in schema
  /// \param[in, out] row Tensor row to push to
  /// \return Status The status code returned
  Status LoadFloatTensor(const nlohmann::json &json_obj, uint32_t col_num, TensorRow *row);

  /// \brief Load int value to tensor row
  /// \param[in] json_obj Json object containing int
  /// \param[in] col_num Column num in schema
  /// \param[in, out] row Tensor row to push to
  /// \return Status The status code returned
  Status LoadIntTensor(const nlohmann::json &json_obj, uint32_t col_num, TensorRow *row);

  /// \brief Load empty tensor to tensor row
  /// \param[in] col_num Column num in schema
  /// \param[in, out] row Tensor row to push to
  /// \return Status The status code returned
  Status LoadEmptyTensor(uint32_t col_num, TensorRow *row);

  /// \brief Load id from file name to tensor row
  /// \param[in] file The file name to get ID from
  /// \param[in] col_num Column num in schema
  /// \param[in, out] row Tensor row to push to
  /// \return Status The status code returned
  Status LoadIDTensor(const std::string &file, uint32_t col_num, TensorRow *row);

  /// \brief Load a tensor row according to a json file
  /// \param[in] row_id_type row_id - id for this tensor row
  /// \param[in] ImageColumns file Json file location
  /// \param[in, out] TensorRow row Json content stored into a tensor row
  /// \return Status The status code returned
  Status LoadTensorRow(row_id_type row_id, const std::string &file, TensorRow *row);

  /// \brief Load a tensor column according to a json file
  /// \param[in] ImageColumns file Json file location
  /// \param[in] index - certain column index
  /// \param[in] js - json object
  /// \param[in, out] TensorRow row Json content stored into a tensor row
  /// \return Status The status code returned
  Status loadColumnData(const std::string &file, int32_t index, nlohmann::json js, TensorRow *row);

  /// \param[in] const std::vector<int64_t> &keys Keys in ioblock
  /// \param[in, out] std::unique_ptr<DataBuffer> db Databuffer to push to
  /// \return Status The status code returned
  Status LoadBuffer(const std::vector<int64_t> &keys, std::unique_ptr<DataBuffer> *db);

  /// \brief Called first when function is called
  /// \return Status The status code returned
  Status LaunchThreadsAndInitOp();

  /// \brief reset Op
  /// \return Status The status code returned
  Status Reset() override;

  /// \brief Gets the next row
  /// \param row[out] - Fetched TensorRow
  /// \return Status The status code returned
  Status GetNextRow(TensorRow *const row) override;

  // Private function for computing the assignment of the column name map.
  // @return Status The status code returned
  Status ComputeColMap() override;

  int32_t rows_per_buffer_;
  std::string folder_path_;  // directory of image folder
  bool decode_;
  std::set<std::string> extensions_;  // extensions allowed
  std::unordered_map<std::string, int32_t> col_name_map_;
  std::unique_ptr<DataSchema> data_schema_;
  int64_t row_cnt_;
  int64_t buf_cnt_;
  int64_t sampler_ind_;
  int64_t dirname_offset_;
  std::vector<std::string> image_rows_;
  TensorPtr sample_ids_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_ALBUM_OP_H_
