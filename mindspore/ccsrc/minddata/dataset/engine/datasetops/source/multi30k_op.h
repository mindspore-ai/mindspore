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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_MULTI30K_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_MULTI30K_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/text_file_op.h"

namespace mindspore {
namespace dataset {
class JaggedConnector;
using StringIndex = AutoIndexObj<std::string>;

class Multi30kOp : public TextFileOp {
 public:
  /// \brief Constructor of Multi30kOp
  /// \note The builder class should be used to call this constructor.
  /// \param[in] num_workers Number of worker threads reading data from multi30k_file files.
  /// \param[in] num_samples Number of rows to read.
  /// \param[in] language_pair List containing text and translation language.
  /// \param[in] worker_connector_size List of filepaths for the dataset files.
  /// \param[in] schema The data schema object.
  /// \param[in] text_files_list File path of multi30k files.
  /// \param[in] op_connector_size Size of each queue in the connector that the child operator pulls from.
  /// \param[in] shuffle_files Whether or not to shuffle the files before reading data.
  /// \param[in] num_devices Shards of data.
  /// \param[in] device_id The device ID within num_devices.
  Multi30kOp(int32_t num_workers, int64_t num_samples, const std::vector<std::string> &language_pair,
             int32_t worker_connector_size, std::unique_ptr<DataSchema> schema,
             const std::vector<std::string> &text_files_list, int32_t op_connector_size, bool shuffle_files,
             int32_t num_devices, int32_t device_id);

  /// \Default destructor.
  ~Multi30kOp() = default;

  /// \brief A print method typically used for debugging.
  /// \param[in] out The output stream to write output to.
  /// \param[in] show_all A bool to control if you want to show all info or just a summary.
  void Print(std::ostream &out, bool show_all);

  /// \brief Return the name of Operator.
  /// \return Status - return the name of Operator.
  std::string Name() const override { return "Multi30kOp"; }

  /// \brief DatasetName name getter.
  /// \param[in] upper If true, the return value is uppercase, otherwise, it is lowercase.
  /// \return std::string DatasetName of the current Op.
  std::string DatasetName(bool upper = false) const { return upper ? "Multi30k" : "multi30k"; }

 private:
  /// \brief Load data into Tensor.
  /// \param[in] line Data read from files.
  /// \param[in] out_row Output tensor.
  /// \param[in] index The index of Tensor.
  Status LoadTensor(const std::string &line, TensorRow *out_row, size_t index);

  /// \brief Read data from files.
  /// \param[in] file_en The paths of multi30k dataset files.
  /// \param[in] start_offset The location of reading start.
  /// \param[in] end_offset The location of reading finished.
  /// \param[in] worker_id The id of the worker that is executing this function.
  Status LoadFile(const std::string &file_en, int64_t start_offset, int64_t end_offset, int32_t worker_id);

  std::vector<std::string> language_pair_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_MULTI30K_OP_H_
