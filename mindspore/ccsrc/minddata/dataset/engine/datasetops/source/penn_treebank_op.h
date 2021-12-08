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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_PENN_TREEBANK_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_PENN_TREEBANK_OP_H_

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/text_file_op.h"
#include "minddata/dataset/util/queue.h"

namespace mindspore {
namespace dataset {
class JaggedConnector;

class PennTreebankOp : public TextFileOp {
 public:
  /// \brief Constructor.
  /// \param[in] num_workers Number of workers reading images in parallel
  /// \param[in] num_samples The number of samples to be included in the dataset.
  /// \param[in] worker_connector_size Size of each internal queue.
  /// \param[in] data_schema Path to dataset schema file.
  /// \param[in] file_list List of files to be read to search for a pattern of files. The list
  ///     will be sorted in a lexicographical order.
  /// \param[in] op_connector_size Size of each queue in the connector that the child operator pulls from.
  /// \param[in] shuffle_files Whether or not to shuffle the files before reading data.
  /// \param[in] num_devices Number of devices that the dataset should be divided into.
  /// \param[in] device_id The device ID within num_devices. This argument should be
  ///     specified only when num_devices is also specified.
  PennTreebankOp(int32_t num_workers, int64_t num_samples, int32_t worker_connector_size, std::unique_ptr<DataSchema>,
                 const std::vector<std::string> &file_list, int32_t op_connector_size, bool shuffle_files,
                 int32_t num_devices, int32_t device_id);

  /// \brief Default destructor.
  ~PennTreebankOp() = default;

  /// \brief A print method typically used for debugging.
  /// \param[in] out he output stream to write output to.
  /// \param[in] show_all A bool to control if you want to show all info or just a summary.
  void Print(std::ostream &out, bool show_all) const override;

  /// \brief Op name getter.
  /// \return Name of the current Op.
  std::string Name() const override { return "PennTreebankOp"; }

  /// \brief DatasetName name getter.
  /// \return DatasetName of the current Op.
  std::string DatasetName(bool upper = false) const { return upper ? "PennTreebank" : "penn treebank"; }
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_PENN_TREEBANK_OP_H_
