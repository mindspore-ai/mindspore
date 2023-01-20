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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SPEECH_COMMANDS_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SPEECH_COMMANDS_OP_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/engine/datasetops/source/mappable_leaf_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/util/services.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/wait_post.h"

namespace mindspore {
namespace dataset {
class SpeechCommandsOp : public MappableLeafOp {
 public:
  /// Constructor.
  /// \param[in] std::string - dataset_dir - directory of SpeechCommands dataset.
  /// \param[in] std::string - usage - directory of SpeechCommands dataset.
  /// \param[in] uint32_t - num_workers - Num of workers reading audios in parallel.
  /// \param[in] uint32_t - queue_size - connector queue size.
  /// \param[in] std::unique_ptr<DataSchema> - data_schema - data schema of SpeechCommands dataset.
  /// \param[in] std::unique_ptr<Sampler> - sampler - sampler tells SpeechCommands what to read.
  SpeechCommandsOp(const std::string &dataset_dir, const std::string &usage, int32_t num_workers, int32_t queue_size,
                   std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler);

  /// Destructor.
  ~SpeechCommandsOp() override = default;

  /// A print method typically used for debugging.
  /// \param[out] out - out stream.
  /// \param[in] show_all - whether to show all information.
  void Print(std::ostream &out, bool show_all) const override;

  /// Function to count the number of samples in the SpeechCommands dataset.
  /// \param[in] num_rows output arg that will hold the actual dataset size.
  /// \return Status - The status code returned.
  Status CountTotalRows(int64_t *num_rows);

  /// Op name getter.
  /// \return Name of the current Op.
  std::string Name() const override { return "SpeechCommandsOp"; }

 private:
  /// Load a tensor row.
  /// \param[in] row_id - row id.
  /// \param[in] trow - waveform & sample_rate & label & speaker_id & utterance_number
  ///     read into this tensor row.
  /// \return Status - The status code returned.
  Status LoadTensorRow(row_id_type row_id, TensorRow *trow) override;

  /// \param[in] pf_path - the real path of root directory.
  /// \param[in] pf_usage - usage.
  /// \return Status - The status code returned.
  Status ParseFileList(const std::string &pf_path, const std::string &pf_usage);

  /// Called first when function is called.
  /// \return Status - The status code returned.
  Status PrepareData();

  /// Walk all folders to read all ".wav" files.
  /// \param[in] walk_path - real path to traverse.
  /// \return Status - The status code returned.
  Status WalkAllFiles(const std::string &walk_path);

  /// Get detail info of wave filename by regex.
  /// \param[in] file_path - wave file path.
  /// \param[out] label - label.
  /// \param[out] speaker_id - speaker id.
  /// \param[out] utterance_number - utterance number.
  /// \return Status - The status code returned.
  Status GetFileInfo(const std::string &file_path, std::string *label, std::string *speaker_id,
                     int32_t *utterance_number);

  // Private function for computing the assignment of the column name map.
  /// \return Status - The status code returned.
  Status ComputeColMap() override;

  std::string dataset_dir_;
  std::string usage_;  // can only be "test", "train", "valid" or "all".
  std::unique_ptr<DataSchema> data_schema_;

  std::set<std::string> all_wave_files;         // all wave files in dataset_dir.
  std::set<std::string> loaded_names;           // loaded file names from txt files.
  std::vector<std::string> selected_files_vec;  // vector of filenames for sequential loading.

  std::mutex mux_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SPEECH_COMMANDS_OP_H_
