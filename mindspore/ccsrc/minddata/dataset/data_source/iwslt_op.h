/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_IWSLT_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_IWSLT_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "./tinyxml2.h"

#include "include/utils/common.h"
#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/data_source/io_block.h"
#include "minddata/dataset/data_source/nonmappable_leaf_op.h"
#include "minddata/dataset/engine/jagged_connector.h"

using tinyxml2::XMLDocument;
using tinyxml2::XMLElement;
using tinyxml2::XMLError;

namespace mindspore {
namespace dataset {
class JaggedConnector;

/// \class IWSLTOp.
/// \brief A Op derived class to represent IWSLT Op.
class IWSLTOp : public NonMappableLeafOp {
 public:
  enum IWSLTType { kIWSLT2016, kIWSLT2017 };

  /// \brief Constructor of IWSLTOp.
  /// \param[in] num_workers Number of worker threads reading data from yelp_review files.
  /// \param[in] num_samples The number of samples to be included in the dataset.
  /// \param[in] worker_connector_size Size of each internal queue.
  /// \param[in] op_connector_size Size of each queue in the connector that the child operator pulls from.
  /// \param[in] shuffle_files Whether or not to shuffle the files before reading data.
  /// \param[in] num_devices Number of devices that the dataset should be divided into.
  /// \param[in] device_id The device ID within num_devices.
  /// \param[in] data_schema Schema of dataset.
  /// \param[in] type Type of data set read.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Usage of this dataset, can be "train", "test", "valid" or "all" data.
  /// \param[in] language_pair List containing src and tgt language.
  /// \param[in] valid_set A string to identify validation set.
  /// \param[in] test_set A string to identify test set.
  IWSLTOp(int32_t num_workers, int64_t num_samples, int32_t worker_connector_size, int32_t op_connector_size,
          bool shuffle_files, int32_t num_devices, int32_t device_id, std::unique_ptr<DataSchema>, IWSLTType type,
          const std::string &dataset_dir, const std::string &usage, const std::vector<std::string> &language_pair,
          const std::string &valid_set, const std::string &test_set);

  /// \brief Destructor.
  ~IWSLTOp() = default;

  /// \brief A print method typically used for debugging.
  /// \param[out] out The output stream to write output to.
  /// \param[in] show_all A bool to control if you want to show all info or just a summary.
  void Print(std::ostream &out, bool show_all) const override;

  /// \brief Instantiates the internal queues and connectors.
  /// \return Status The error code returned.
  Status Init() override;

  /// \brief Function to count the number of samples in the IWSLT dataset.
  /// \param[in] type IWSLT data set version, which can be kIWSLT2016 and kIWSLT2017.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of IWSLT2017, can be "train", "valid", "test" or "all".
  /// \param[in] language_pair List containing src and tgt language.
  /// \param[in] valid_set A string to identify validation set.
  /// \param[in] test_set A string to identify test set.
  /// \param[out] count The total number of rows in file.
  /// \return Status The status code returned.
  static Status CountTotalRows(IWSLTType type, const std::string &dataset_dir, const std::string &usage,
                               const std::vector<std::string> &language_pair, const std::string &valid_set,
                               const std::string &test_set, int64_t *count);

  /// \brief Op name getter.
  /// \return std::string Name of the current Op.
  std::string Name() const override { return "IWSLTOp"; }

  /// \brief DatasetName name getter.
  /// \param[in] upper If true, the return value is uppercase, otherwise, it is lowercase.
  /// \return std::string DatasetName of the current Op.
  std::string DatasetName(bool upper = false) const { return upper ? "IWSLT" : "iwslt"; }

  // \brief File names getter.
  // \return std::vector<std::string> Vector of the input file names.
  std::vector<std::string> FileNames() { return src_target_file_list_; }

 private:
  /// \brief Split string based on a character delimiter.
  /// \param[in] s The input string.
  /// \param[in] delim The delimiter.
  /// \return std::vector<std::string> The result after segmentation.
  std::vector<std::string> Split(const std::string &s, const std::string &delim);

  /// \brief Remove the characters specified at the beginning and end.
  /// \param[in] text The input string.
  /// \param[in] character The removed character.
  /// \return Status The status code returned.
  Status Trim(std::string *text, const std::string &character);

  /// \brief Function to count the number of samples in one data file.
  /// \param[in] file Path to the data file.
  /// \return int64_t The total number of rows in file.
  int64_t CountFileRows(const std::string &file);

  /// \brief Parses a single row and puts the data into a tensor table.
  /// \param[in] line The content of the row.
  /// \param[out] out_row Output tensor.
  /// \param[in] index The id of the row filled in the tensor table.
  /// \return Status The status code returned.
  Status LoadTensor(const std::string &line, TensorRow *out_row, size_t index);

  /// \brief Reads a IWSLT file and loads the data into multiple TensorRows.
  /// \param[in] file The file to read.
  /// \param[in] start_offset The start offset of file.
  /// \param[in] end_offset The end offset of file.
  /// \param[in] worker_id The id of the worker that is executing this function.
  /// \return Status The status code returned.
  Status LoadFile(const std::string &file, int64_t start_offset, int64_t end_offset, int32_t worker_id) override;

  /// \brief Fill the IOBlockQueue.
  /// \param[in] i_keys Keys of file to fill to the IOBlockQueue.
  /// \return Status The status code returned.
  Status FillIOBlockQueue(const std::vector<int64_t> &i_keys) override;

  /// \brief Calculate number of rows in each shard.
  /// \return Status The status code returned.
  Status CalculateNumRowsPerShard() override;

  /// \brief Private function for computing the assignment of the column name map.
  /// \return Status The status code returned.
  Status ComputeColMap() override;

  /// \brief Write the data of the source file and the target file to a new file after cleaning.
  /// \param[in] src_file_path Source file path.
  /// \param[in] target_file_path Target file path.
  /// \param[in] new_file_path Write data to new file path.
  /// \return Status The status code returned.
  Status CleanXmlFile(const std::string &src_file_path, const std::string &target_file_path,
                      const std::string &new_file_path);

  /// \brief Determine whether the centent contains the specified label.
  /// \param[in] content This content to be determined.
  /// \return bool If it contains, return true, otherwise, return false.
  bool IsContainTags(const std::string &content);

  /// \brief Write the data of the source file and the target file to a new file after cleaning.
  /// \param[in] src_file_path Source file path.
  /// \param[in] target_file_path Target file path.
  /// \param[in] new_file_path Write data to new file path.
  /// \return Status The status code returned.
  Status CleanTagFile(const std::string &file_path, const std::string &target_file_path,
                      const std::string &new_file_path);

  // \brief Get all files in the dataset_dir_.
  // \return Status The status code returned.
  Status GetFiles();

  /// \brief Generate IWSLT2016 training data set file list.
  /// \param[in] dir The directory where the files are stored.
  /// \param[in] src_language The source language type.
  /// \param[in] target_language The target language type.
  /// \param[in] suffix The file suffix.
  /// \return std::string The file path.
  std::string GenerateIWSLT2016TagsFileName(Path dir, const std::string &src_language,
                                            const std::string &target_language, const std::string &suffix);

  /// \brief Generate IWSLT2016 valid data set or test data set file list.
  /// \param[in] dir The directory where the files are stored.
  /// \param[in] src_language The source language type.
  /// \param[in] target_language The target language type.
  /// \param[in] set_type The type of data set read.
  /// \param[in] suffix The file suffix.
  /// \return std::string The file path.
  std::string GenerateIWSLT2016XMLFileName(Path dir, const std::string &src_language,
                                           const std::string &target_language, const std::string &set_type,
                                           const std::string &suffix);

  /// \brief Generate IWSLT2017 training data set file list.
  /// \param[in] dir The directory where the files are stored.
  /// \param[in] src_language The source language type.
  /// \param[in] target_language The target language type.
  /// \param[in] suffix The file suffix.
  /// \return std::string The file path.
  std::string GenerateIWSLT2017TagsFileName(Path dir, const std::string &src_language,
                                            const std::string &target_language, const std::string &suffix);

  /// \brief Generate IWSLT2016 valid data set or test data set file list.
  /// \param[in] dir The directory where the files are stored.
  /// \param[in] src_language The source language type.
  /// \param[in] target_language The target language type.
  /// \param[in] set_type The type of data set read.
  /// \param[in] suffix The file suffix.
  /// \return std::string The file path.
  std::string GenerateIWSLT2017XMLFileName(Path dir, const std::string &src_language,
                                           const std::string &target_language, const std::string &set_type,
                                           const std::string &suffix);

  /// \brief Generate new file path and write data.
  /// \param[in] src_path_list The source file path.
  /// \param[in] target_path_list The target file path.
  /// \param[out] src_target_file_list The newly generated file path list.
  /// \return Status The status code returned.
  Status GenerateNewFile(const std::vector<std::string> &src_file_list,
                         const std::vector<std::string> &target_file_list,
                         std::vector<std::string> *src_target_file_list);

  IWSLTType iwslt_type_;
  std::unique_ptr<DataSchema> data_schema_;
  std::vector<std::string> src_target_file_list_;
  std::string dataset_dir_;
  std::string usage_;
  std::vector<std::string> language_pair_;
  std::string valid_set_;
  std::string test_set_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_IWSLT_OP_H_
