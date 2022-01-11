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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_EN_WIK9_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_EN_WIK9_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/text_file_op.h"

namespace mindspore {
namespace dataset {
class EnWik9Op : public TextFileOp {
 public:
  /// \brief Constructor.
  /// \param[in] num_workers The number of worker threads reading data from enwiki files.
  /// \param[in] total_rows The number of rows to read.
  /// \param[in] worker_connector_size Size of each internal queue.
  /// \param[in] data_schema The data schema object.
  /// \param[in] files_list List of file paths for the dataset files.
  /// \param[in] op_connector_size Size of each queue in the connector that the child operator pulls from.
  /// \param[in] shuffle_files Whether or not to shuffle the files before reading data.
  /// \param[in] num_devices The number of devices.
  /// \param[in] device_id Id of device.
  EnWik9Op(int32_t num_workers, int64_t total_rows, int32_t worker_connector_size,
           std::unique_ptr<DataSchema> data_schema, const std::vector<std::string> &file_list,
           int32_t op_connector_size, bool shuffle_files, int32_t num_devices, int32_t device_id);

  /// \brief Default destructor.
  ~EnWik9Op() = default;

  /// \brief A print method typically used for debugging.
  /// \param[out] out The output stream to write output to.
  /// \param[in] show_all A bool to control if you want to show all info or just a summary.
  void Print(std::ostream &out, bool show_all) const override;

  /// \brief Op name getter.
  /// \return Name of the current Op.
  std::string Name() const override { return "EnWik9Op"; }

  /// \brief DatasetName name getter.
  /// \param[in] upper A bool to control if you need upper DatasetName.
  /// \return DatasetName of the current Op.
  std::string DatasetName(bool upper = false) const override { return upper ? "EnWik9" : "enwik9"; }

  /// \brief Reads a text file and loads the data into multiple TensorRows.
  /// \param[in] file The file to read.
  /// \param[in] start_offset - the start offset of file.
  /// \param[in] end_offset - the end offset of file.
  /// \param[in] The id of the worker that is executing this function.
  /// \return Status The error code returned.
  Status LoadFile(const std::string &file, int64_t start_offset, int64_t end_offset, int32_t worker_id) override;

 private:
  /// \brief Count number of rows in each file.
  /// \param[in] file Txt file name.
  /// \return int64_t The total number of rows in file.
  int64_t CountTotalRows(const std::string &file) override;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_EN_WIK9_OP_H_
