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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_LIBRI_TTS_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_LIBRI_TTS_OP_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/engine/datasetops/source/mappable_leaf_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/wait_post.h"

namespace mindspore {
namespace dataset {
struct LibriTTSLabelTuple {
  std::string usage;
  std::string utterance_id;
  std::string original_text;
  std::string normalized_text;
  uint32_t speaker_id;
  uint32_t chapter_id;
  std::string label_path;
};

class LibriTTSOp : public MappableLeafOp {
 public:
  /// \brief Constructor.
  /// \param[in] dataset_dir Dir directory of LibriTTS.
  /// \param[in] usage usage of this dataset, can be "dev-clean", "dev-other", "test-clean", "test-other",
  ///     "train-clean-100", "train-clean-360", "train-other-500", or "all".
  /// \param[in] num_workers Number of workers reading audios in parallel.
  /// \param[in] queue_size Connector queue size.
  /// \param[in] data_schema The schema of the LibriTTS dataset.
  /// \param[in] sampler Sampler tells LibriSpeechOp what to read.
  LibriTTSOp(const std::string &dataset_dir, const std::string &usage, int32_t num_workers, int32_t queue_size,
             std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler);

  /// \brief Destructor.
  ~LibriTTSOp() = default;

  /// \brief A print method typically used for debugging.
  /// \param[out] out Output stream.
  /// \param[in] show_all Whether to show all information.
  void Print(std::ostream &out, bool show_all) const override;

  /// \brief Function to count the number of samples in the LibriTTS dataset.
  /// \param[in] dir Path to the LibriTTS directory.
  /// \param[in] usage Select the data set section.
  /// \param[out] count Output arg that will hold the minimum of the actual dataset size and numSamples.
  /// \return Status The status code returned.
  static Status CountTotalRows(const std::string &dir, const std::string &usage, int64_t *count);

  /// \brief Op name getter.
  /// \return Name of the current Op.
  std::string Name() const override { return "LibriTTSOp"; }

 private:
  /// \brief Load a tensor row according to a pair.
  /// \param[in] row_id Id for this tensor row.
  /// \param[out] row Audio & label read into this tensor row.
  /// \return Status The status code returned.
  Status LoadTensorRow(row_id_type row_id, TensorRow *row) override;

  /// \brief Read all paths in the directory.
  /// \param[in] dir File path to be traversed.
  /// \return Status The status code returned.
  Status GetPaths(Path *dir);

  /// \brief Read all label files.
  /// \return Status The status code returned.
  Status GetLabels();

  /// \brief Parse a single wav file.
  /// \param[in] audio_dir Audio file path.
  /// \param[out] waveform The output waveform tensor.
  /// \return Status The status code returned.
  Status ReadAudio(const std::string &audio_dir, std::shared_ptr<Tensor> *waveform);

  /// \brief Prepare all data in the directory.
  /// \return Status The status code returned.
  Status PrepareData();

  /// \brief Private function for computing the assignment of the column name map.
  /// \return Status The status code returned.
  Status ComputeColMap() override;

  const std::string usage_;
  std::string cur_usage_;
  std::string real_path_;
  std::string dataset_dir_;
  std::unique_ptr<DataSchema> data_schema_;
  std::vector<LibriTTSLabelTuple> audio_label_tuples_;
  std::vector<std::string> label_files_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_LIBRI_TTS_OP_H_
