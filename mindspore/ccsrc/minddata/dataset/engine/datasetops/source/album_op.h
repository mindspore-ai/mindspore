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

#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/engine/datasetops/source/mappable_leaf_op.h"
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
class AlbumOp : public MappableLeafOp {
 public:
  /// \brief Constructor
  /// \param[in] num_wkrs - Num of workers reading images in parallel
  /// \param[in] file_dir - directory of Album
  /// \param[in] queue_size - connector size
  /// \param[in] do_decode - decode image files
  /// \param[in] exts - set of file extensions to read, if empty, read everything under the dir
  /// \param[in] data_schema - schema of dataset
  /// \param[in] sampler - sampler tells AlbumOp what to read
  AlbumOp(int32_t num_wkrs, std::string file_dir, int32_t queue_size, bool do_decode, const std::set<std::string> &exts,
          std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler);

  /// \brief Destructor.
  ~AlbumOp() = default;

  /// \brief Initialize AlbumOp related var, calls the function to walk all files
  /// \return Status The status code returned
  Status PrescanEntry();

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
  /// \brief Load image to tensor row
  /// \param[in] image_file Image name of file
  /// \param[in] col_num Column num in schema
  /// \param[in, out] row Tensor row to push to
  /// \return Status The status code returned
  Status LoadImageTensor(const std::string &image_file, int32_t col_num, TensorRow *row);

  /// \brief Load vector of ints to tensor, append tensor to tensor row
  /// \param[in] json_obj Json object containing multi-dimensional label
  /// \param[in] col_num Column num in schema
  /// \param[in, out] row Tensor row to push to
  /// \return Status The status code returned
  Status LoadIntArrayTensor(const nlohmann::json &json_obj, int32_t col_num, TensorRow *row);

  /// \brief Load vector of floatss to tensor, append tensor to tensor row
  /// \param[in] json_obj Json object containing array data
  /// \param[in] col_num Column num in schema
  /// \param[in, out] row Tensor row to push to
  /// \return Status The status code returned
  Status LoadFloatArrayTensor(const nlohmann::json &json_obj, int32_t col_num, TensorRow *row);

  /// \brief Load string array into a tensor, append tensor to tensor row
  /// \param[in] json_obj Json object containing string tensor
  /// \param[in] col_num Column num in schema
  /// \param[in, out] row Tensor row to push to
  /// \return Status The status code returned
  Status LoadStringArrayTensor(const nlohmann::json &json_obj, int32_t col_num, TensorRow *row);

  /// \brief Load string into a tensor, append tensor to tensor row
  /// \param[in] json_obj Json object containing string tensor
  /// \param[in] col_num Column num in schema
  /// \param[in, out] row Tensor row to push to
  /// \return Status The status code returned
  Status LoadStringTensor(const nlohmann::json &json_obj, int32_t col_num, TensorRow *row);

  /// \brief Load float value to tensor row
  /// \param[in] json_obj Json object containing float
  /// \param[in] col_num Column num in schema
  /// \param[in, out] row Tensor row to push to
  /// \return Status The status code returned
  Status LoadFloatTensor(const nlohmann::json &json_obj, int32_t col_num, TensorRow *row);

  /// \brief Load int value to tensor row
  /// \param[in] json_obj Json object containing int
  /// \param[in] col_num Column num in schema
  /// \param[in, out] row Tensor row to push to
  /// \return Status The status code returned
  Status LoadIntTensor(const nlohmann::json &json_obj, int32_t col_num, TensorRow *row);

  /// \brief Load empty tensor to tensor row
  /// \param[in] col_num Column num in schema
  /// \param[in, out] row Tensor row to push to
  /// \return Status The status code returned
  Status LoadEmptyTensor(int32_t col_num, TensorRow *row);

  /// \brief Load id from file name to tensor row
  /// \param[in] file The file name to get ID from
  /// \param[in] col_num Column num in schema
  /// \param[in, out] row Tensor row to push to
  /// \return Status The status code returned
  Status LoadIDTensor(const std::string &file, int32_t col_num, TensorRow *row);

  /// \brief Load a tensor row according to a json file
  /// \param[in] row_id_type row_id - id for this tensor row
  /// \param[in, out] TensorRow row Json content stored into a tensor row
  /// \return Status The status code returned
  Status LoadTensorRow(row_id_type row_id, TensorRow *row) override;

  /// \brief Load a tensor column according to a json file
  /// \param[in] ImageColumns file Json file location
  /// \param[in] index - certain column index
  /// \param[in] js - json object
  /// \param[in, out] TensorRow row Json content stored into a tensor row
  /// \return Status The status code returned
  Status loadColumnData(const std::string &file, int32_t index, nlohmann::json js, TensorRow *row);

  /// \brief Called first when function is called
  /// \return Status The status code returned
  Status LaunchThreadsAndInitOp() override;

  /// \brief Gets the next row
  /// \param row[out] - Fetched TensorRow
  /// \return Status The status code returned
  Status GetNextRowPullMode(TensorRow *const row) override;

  /// Private function for computing the assignment of the column name map.
  /// \return Status The status code returned
  Status ComputeColMap() override;

  std::string folder_path_;  // directory of image folder
  bool decode_;
  std::set<std::string> extensions_;  // extensions allowed
  std::unordered_map<std::string, int32_t> col_name_map_;
  std::unique_ptr<DataSchema> data_schema_;
  int64_t sampler_ind_;
  int64_t dirname_offset_;
  std::vector<std::string> image_rows_;
  TensorPtr sample_ids_;

  uint32_t curr_row_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_ALBUM_OP_H_
