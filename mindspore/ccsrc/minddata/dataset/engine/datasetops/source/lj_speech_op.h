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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_LJ_SPEECH_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_LJ_SPEECH_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/core/tensor.h"

#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/source/mappable_leaf_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/util/services.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/wait_post.h"

namespace mindspore {
namespace dataset {
/// \brief Read LJSpeech dataset.
class LJSpeechOp : public MappableLeafOp {
 public:
  /// \brief Constructor.
  /// \param[in] file_dir Directory of lj_speech dataset.
  /// \param[in] num_workers Number of workers reading audios in parallel.
  /// \param[in] queue_size Connector queue size.
  /// \param[in] data_schema Data schema of lj_speech dataset.
  /// \param[in] sampler Sampler tells LJSpeechOp what to read.
  LJSpeechOp(const std::string &file_dir, int32_t num_workers, int32_t queue_size,
             std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler);

  /// \brief Destructor.
  ~LJSpeechOp() = default;

  /// \brief A print method typically used for debugging.
  /// \param[out] out The output stream to write output to.
  /// \param[in] show_all A bool to control if you want to show all info or just a summary.
  void Print(std::ostream &out, bool show_all) const override;

  /// \brief Function to count the number of samples in the LJSpeech dataset.
  /// \param[in] dir Path to the directory of LJSpeech dataset.
  /// \param[out] count Output arg that will hold the actual dataset size.
  /// \return Status
  static Status CountTotalRows(const std::string &dir, int64_t *count);

  /// \brief Op name getter.
  /// \return Name of the current Op.
  std::string Name() const override { return "LJSpeechOp"; }

 protected:
  /// \brief Called first when function is called.
  /// \return Status
  Status PrepareData() override;

 private:
  /// \brief Load a tensor row.
  /// \param[in] index Index need to load.
  /// \param[out] trow Waveform & sample_rate & transcription & normalized_transcription read into this tensor row.
  /// \return Status the status code returned.
  Status LoadTensorRow(row_id_type index, TensorRow *trow) override;

  /// \brief Private function for computing the assignment of the column name map.
  /// \return Status
  Status ComputeColMap() override;

  std::string folder_path_;
  std::unique_ptr<DataSchema> data_schema_;
  std::vector<std::vector<std::string>> meta_info_list_;  // the shape is (N, 3)
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_LJ_SPEECH_OP_H_
