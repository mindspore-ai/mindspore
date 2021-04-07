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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_ALBUM_ANDROID_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_ALBUM_ANDROID_OP_H_

#include <deque>
#include <memory>
#include <queue>
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <utility>
#include <vector>
#include <unordered_map>
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/data_buffer.h"
#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
// Forward declares
template <typename T>
class Queue;

// Define row information as a list of file objects to read
using FolderImages = std::shared_ptr<std::pair<std::string, std::queue<std::string>>>;

/// \class AlbumOp
class AlbumOp {
 public:
  /// \brief Constructor
  /// \param[in] file_dir - directory of Album
  /// \param[in] do_decode - decode image files
  /// \param[in] schema_file - schema file
  /// \param[in] column_names - column name
  /// \param[in] exts - set of file extensions to read, if empty, read everything under the dir
  AlbumOp(const std::string &file_dir, bool do_decode, const std::string &schema_file,
          const std::vector<std::string> &column_names, const std::set<std::string> &exts);

  /// \brief Constructor
  /// \param[in] file_dir - directory of Album
  /// \param[in] do_decode - decode image files
  /// \param[in] schema_file - schema file
  /// \param[in] column_names - column name
  /// \param[in] exts - set of file extensions to read, if empty, read everything under the dir
  /// \param[in] index - the specific file index
  AlbumOp(const std::string &file_dir, bool do_decode, const std::string &schema_file,
          const std::vector<std::string> &column_names, const std::set<std::string> &exts, uint32_t index);

  /// \brief Destructor.
  ~AlbumOp() = default;

  /// \brief Initialize AlbumOp related var, calls the function to walk all files
  /// \return - The error code returned
  Status PrescanEntry();

  /// \brief Initialize AlbumOp related var, calls the function to walk all files
  /// \return - The error code returned
  bool GetNextRow(std::unordered_map<std::string, std::shared_ptr<Tensor>> *map_row);

  /// \brief Check if image ia valid.Only support JPEG/PNG/GIF/BMP
  ///     This function could be optimized to return the tensor to reduce open/closing files
  /// \return bool - if file is bad then return false
  bool CheckImageType(const std::string &file_name, bool *valid);

  /// \brief Name of the current Op
  /// @return op name
  std::string Name() const { return "AlbumOp"; }

  /// \brief disable rotate
  // @return
  void DisableRotate() { this->rotate_ = false; }

 private:
  /// \brief Load image to tensor
  /// \param[in] image_file Image name of file
  /// \param[in] col_num Column num in schema
  /// \param[in,out] Tensor to push to
  /// \return Status The error code returned
  Status LoadImageTensor(const std::string &image_file, uint32_t col_num, TensorPtr *tensor);

  /// \brief Load vector of ints to tensor, append tensor to tensor
  /// \param[in] json_obj Json object containing multi-dimensional label
  /// \param[in] col_num Column num in schema
  /// \param[in,out] Tensor to push to
  /// \return Status The error code returned
  Status LoadIntArrayTensor(const nlohmann::json &json_obj, uint32_t col_num, TensorPtr *tensor);

  /// \brief Load vector of floatss to tensor, append tensor to tensor
  /// \param[in] json_obj Json object containing array data
  /// \param[in] col_num Column num in schema
  /// \param[in,out] Tensor to push to
  /// \return Status The error code returned
  Status LoadFloatArrayTensor(const nlohmann::json &json_obj, uint32_t col_num, TensorPtr *tensor);

  /// \brief Load string array into a tensor, append tensor to tensor
  /// \param[in] json_obj Json object containing string tensor
  /// \param[in] col_num Column num in schema
  /// \param[in,out] Tensor to push to
  /// \return Status The error code returned
  Status LoadStringArrayTensor(const nlohmann::json &json_obj, uint32_t col_num, TensorPtr *tensor);

  /// \brief Load string into a tensor, append tensor to tensor
  /// \param[in] json_obj Json object containing string tensor
  /// \param[in] col_num Column num in schema
  /// \param[in,out]  Tensor to push to
  /// \return Status The error code returned
  Status LoadStringTensor(const nlohmann::json &json_obj, uint32_t col_num, TensorPtr *tensor);

  /// \brief Load float value to tensor
  /// \param[in] json_obj Json object containing float
  /// \param[in] col_num Column num in schema
  /// \param[in,out]  Tensor to push to
  /// \return Status The error code returned
  Status LoadFloatTensor(const nlohmann::json &json_obj, uint32_t col_num, TensorPtr *tensor);

  /// \brief Load int value to tensor
  /// \param[in] json_obj Json object containing int
  /// \param[in] col_num Column num in schema
  /// \param[in,out] Tensor to push to
  /// \return Status The error code returned
  Status LoadIntTensor(const nlohmann::json &json_obj, uint32_t col_num, TensorPtr *tensor);

  /// \brief Load empty tensor to tensor
  /// \param[in] col_num Column num in schema
  /// \param[in,out] Tensor to push to
  /// \return Status The error code returned
  Status LoadEmptyTensor(uint32_t col_num, TensorPtr *tensor);

  /// \brief Load id from file name to tensor
  /// \param[in] file The file name to get ID from
  /// \param[in] col_num Column num in schema
  /// \param[in,out] Tensor to push to
  /// \return Status The error code returned
  Status LoadIDTensor(const std::string &file, uint32_t col_num, TensorPtr *tensor);

  /// \brief Load a tensor according to a json file
  /// \param[in] row_id_type row_id - id for this tensor row
  /// \param[in] ImageColumns file Json file location
  /// \param[in,out] TensorRow Json content stored into a tensor row
  /// \return Status The error code returned
  Status LoadTensorRow(row_id_type row_id, const std::string &file,
                       std::unordered_map<std::string, std::shared_ptr<Tensor>> *map_row);

  /// \brief get image exif orientation
  /// \param[in] file file path
  int GetOrientation(const std::string &file);

  /// \brief is read column name
  /// \param[in] column_name
  bool IsReadColumn(const std::string &column_name);

  Status LoadTensorRowByIndex(int index, const std::string &file, const nlohmann::json &js,
                              std::unordered_map<std::string, std::shared_ptr<Tensor>> *map_row);

  Status LoadIntTensorRowByIndex(int index, bool is_array, const nlohmann::json &column_value,
                                 std::unordered_map<std::string, std::shared_ptr<Tensor>> *map_row);

  std::string folder_path_;  // directory of image folder
  bool decode_;
  std::vector<std::string> columns_to_load_;
  std::set<std::string> extensions_;  // extensions allowed
  std::unique_ptr<DataSchema> data_schema_;
  std::string schema_file_;
  int64_t row_cnt_;
  int64_t current_cnt_;
  int64_t buf_cnt_;
  int64_t dirname_offset_;
  bool sampler_;
  int64_t sampler_index_;
  std::vector<std::string> image_rows_;
  bool rotate_;
  std::vector<std::string> column_names_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_ALBUM_ANDROID_OP_H_
