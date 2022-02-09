/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SQUAD_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SQUAD_OP_H_

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/engine/datasetops/source/nonmappable_leaf_op.h"
#include "minddata/dataset/engine/jagged_connector.h"

namespace mindspore {
namespace dataset {
/// \class SQuADOp
/// \brief Loading Operator of SQuAD Dataset.
class SQuADOp : public NonMappableLeafOp {
 public:
  /// \brief Constructor of SQuADOp.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of SQuAD.
  /// \param[in] num_workers Number of worker threads reading data from tf_file files.
  /// \param[in] num_samples Number of samples.
  /// \param[in] worker_connector_size Size of each internal queue.
  /// \param[in] schema The data schema object.
  /// \param[in] op_connector_size Size of each queue in the connector that the child operator pulls from.
  /// \param[in] shuffle_files Whether or not to shuffle the files before reading data.
  /// \param[in] num_devices Number of devices.
  /// \param[in] device_id Device id.
  SQuADOp(const std::string &dataset_dir, const std::string &usage, int32_t num_workers, int64_t num_samples,
          int32_t worker_connector_size, std::unique_ptr<DataSchema> schema, int32_t op_connector_size,
          bool shuffle_files, int32_t num_devices, int32_t device_id);

  /// \brief Default destructor of SQuADOp.
  ~SQuADOp() = default;

  /// \brief A print method typically used for debugging.
  /// \param[out] out The output stream to write output to.
  /// \param[in] show_all A bool to control if you want to show all info or just a summary.
  void Print(std::ostream &out, bool show_all) const override;

  /// \brief Instantiates the internal queues and connectors.
  /// \return Status The error code returned.
  Status Init() override;

  /// \brief Get total tensor rows in files.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of SQuAD.
  /// \param[out] count Number of tensor rows.
  /// \return Status The error code returned.
  static Status CountAllFileRows(const std::string &dataset_dir, const std::string &usage, int64_t *count);

  /// \brief File names getter.
  /// \return Vector of the input file names.
  std::vector<std::string> FileNames() { return squad_files_list_; }

  /// \brief Op name getter.
  /// \return Name of the current Op.
  std::string Name() const override { return "SQuADOp"; }

 private:
  /// \brief A single row created from scalar and puts the data into a tensor table.
  /// \param[in] scalar_item The content of the row.
  /// \param[out] out_row The id of the row filled in the tensor table.
  /// \param[in] index The index of the Tecsor Row.
  /// \return Status The error code returned.
  Status LoadTensorFromScalar(const std::string &scalar_item, TensorRow *out_row, size_t index);

  /// \brief A single row created from vector and puts the data into a tensor table.
  /// \param[in] vector_item The content of the row.
  /// \param[out] out_row The id of the row filled in the tensor table.
  /// \param[in] index The index of the Tecsor Row.
  /// \return Status The error code returned.
  template <typename T>
  Status LoadTensorFromVector(const std::vector<T> &vector_item, TensorRow *out_row, size_t index);

  /// \brief Reads a squad file and loads the data into multiple TensorRows.
  /// \param[in] file The file to read.
  /// \param[in] start_offset The start offset of file.
  /// \param[in] end_offset The end offset of file.
  /// \param[in] worker_id The id of the worker that is executing this function.
  /// \return Status The error code returned.
  Status LoadFile(const std::string &file, int64_t start_offset, int64_t end_offset, int32_t worker_id) override;

  /// \brief Fill the IOBlockQueue.
  /// \param[in] i_keys Kys of file to fill to the IOBlockQueue.
  /// \return Status The error code returned.
  Status FillIOBlockQueue(const std::vector<int64_t> &i_keys) override;

  /// \brief Calculate number of rows in each shard.
  /// \return Status The error code returned.
  Status CalculateNumRowsPerShard() override;

  /// \brief Private function for computing the assignment of the column name map.
  /// \return Status The error code returned.
  Status ComputeColMap() override;

  /// \brief Get all files in the dataset_dir_.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of SQuAD.
  /// \param[out] data_files_list The files list in the root directory.
  /// \return Status The status code returned.
  static Status GetFiles(const std::string &dataset_dir, const std::string &usage,
                         std::vector<std::string> *data_files_list);

  /// \brief Search the node in json object.
  /// \param[in] input_tree The json object.
  /// \param[in] node_name The name of the node to search.
  /// \param[out] output_node The output node.
  /// \return Status The error code returned.
  template <typename T>
  static Status SearchNodeInJson(const nlohmann::json &input_tree, std::string node_name, T *output_node);

  /// \brief Count the number of tensorRows in the file.
  /// \param[in] file The SQuAD file name.
  /// \param[out] count The number of tensorRows in the file.
  /// \return Status The error code returned.
  static Status CountTensorRowsPreFile(const std::string &file, int64_t *count);

  /// \brief Load the vector of answer start and answer text.
  /// \param[in] answers_tree Answer list of json.
  /// \param[out] answer_text_vec The vector of answer text.
  /// \param[out] answer_start_vec The vector of answer start.
  /// \return Status The error code returned.
  Status AnswersLoad(const nlohmann::json &answers_tree, std::vector<std::string> *answer_text_vec,
                     std::vector<uint32_t> *answer_start_vec);

  std::string dataset_dir_;
  std::string usage_;
  std::vector<std::string> squad_files_list_;
  std::unique_ptr<DataSchema> data_schema_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SQUAD_OP_H_
