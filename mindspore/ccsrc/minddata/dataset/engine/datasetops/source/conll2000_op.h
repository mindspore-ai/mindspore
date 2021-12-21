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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_CONLL2000_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_CONLL2000_OP_H_

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

class CoNLL2000Op : public TextFileOp {
 public:
  /// \Constructor of CoNLL2000Op.
  CoNLL2000Op(int32_t num_workers, int64_t total_rows, int32_t worker_connector_size, std::unique_ptr<DataSchema>,
              const std::vector<std::string> &conll2000_file_list, int32_t op_connector_size, bool shuffle_files,
              int32_t num_devices, int32_t device_id);

  /// \Default destructor.
  ~CoNLL2000Op() = default;

  /// \brief A print method typically used for debugging.
  /// \param[in] out The output stream to write output to.
  /// \param[in] show_all A bool to control if you want to show all info or just a summary.
  void Print(std::ostream &out, bool show_all) const override;

  /// \brief Op name getter.
  /// \return Name of the current Op.
  std::string Name() const override { return "CoNLL2000Op"; }

  /// \brief brief description DatasetName name getter
  /// \param[in] upper Needs to be capitalized or not
  /// \return DatasetName of the current Op
  std::string DatasetName(bool upper = false) const { return upper ? "CoNLL2000" : "conll2000"; }

 private:
  /// \brief Parses a single row and puts the data into multiple TensorRows.
  /// \param[in] column The content of the column.
  /// \param[in] out_row The tensor table to put the parsed data in.
  /// \param[in] index Serial number of column.
  /// \return Status The error code returned.
  Status LoadTensor(const std::vector<std::string> &column, TensorRow *out_row, size_t index);

  /// \brief Removes excess space before and after the string.
  /// \param[in] str The input string.
  /// \return A string.
  std::string Strip(const std::string &str);

  /// \brief Split string based on a character delimiter.
  /// \param[in] s The input string.
  /// \param[in] delim Symbols for separating string.
  /// \return A string vector.
  std::vector<std::string> Split(const std::string &s, char delim);

  /// \brief Specify that the corresponding data is translated into Tensor.
  /// \param[in] word A list of words in a sentence.
  /// \param[in] pos_tag Pos_tag part of speech.
  /// \param[in] chunk_tag Chunk_tag part of speech.
  /// \param[in] file The file to read.
  /// \param[in] worker_id The id of the worker that is executing this function.
  /// \return Status The error code returned.
  Status Load(const std::vector<std::string> &word, const std::vector<std::string> &pos_tag,
              const std::vector<std::string> &chunk_tag, const std::string &file, int32_t worker_id);

  /// \brief Reads a text file and loads the data into multiple TensorRows.
  /// \param[in] file The file to read.
  /// \param[in] start_offset The start offset of file.
  /// \param[in] end_offset The end offset of file.
  /// \param[in] worker_id The id of the worker that is executing this function.
  /// \return Status The error code returned.
  Status LoadFile(const std::string &file, int64_t start_offset, int64_t end_offset, int32_t worker_id) override;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_CONLL2000_OP_H_
