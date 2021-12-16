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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_YAHOO_ANSWERS_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_YAHOO_ANSWERS_OP_H_

#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/csv_op.h"

namespace mindspore {
namespace dataset {
class YahooAnswersOp : public CsvOp {
 public:
  /// \brief Constructor.
  /// \param[in] dataset_files_list - List of file paths for the dataset files.
  /// \param[in] field_delim - A char that indicates the delimiter to separate fields.
  /// \param[in] column_default - List of default values for the CSV field (default={}). Each item in the list is
  ///     either a valid type (float, int, or string).
  /// \param[in] column_name - List of column names of the dataset file.
  /// \param[in] num_workers - Num of workers reading files in parallel.
  /// \param[in] num_samples - The number of samples to be included in the dataset.
  /// \param[in] worker_connector_size - Size of each internal queue.
  /// \param[in] op_connector_size - Size of each queue in the connector that the child operator pulls from.
  /// \param[in] shuffle_files - Whether or not to shuffle the files before reading data.
  /// \param[in] num_devices - Number of devices that the dataset should be divided into.
  /// \param[in] device_id - The device ID within num_devices.
  YahooAnswersOp(const std::vector<std::string> &dataset_files_list, char field_delim,
                 const std::vector<std::shared_ptr<BaseRecord>> &column_default,
                 const std::vector<std::string> &column_name, int32_t num_workers, int64_t num_samples,
                 int32_t worker_connector_size, int32_t op_connector_size, bool shuffle_files, int32_t num_devices,
                 int32_t device_id);

  /// \brief Destructor.
  ~YahooAnswersOp() = default;

  /// \brief A print method typically used for debugging
  /// \param[out] out The output stream to write output to
  /// \param[in] show_all A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override;

  /// \brief DatasetName name getter.
  /// \param[in] upper A bool to control if you want to return uppercase or lowercase Op name.
  /// \return DatasetName of the current Op.
  std::string DatasetName(bool upper = false) const { return upper ? "YahooAnswers" : "yahoo answers"; }

  /// \brief Op name getter
  /// \return Name of the current Op.
  std::string Name() const override { return "YahooAnswersOp"; }
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_YAHOO_ANSWERS_OP_H_
