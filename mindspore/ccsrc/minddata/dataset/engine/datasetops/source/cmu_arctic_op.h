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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_CMU_ARCTIC_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_CMU_ARCTIC_OP_H_

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
class CMUArcticOp : public MappableLeafOp {
 public:
  /// \brief Constructor.
  /// \param[in] dataset_dir Directory of CMUArctic.
  /// \param[in] name Part of this dataset, can be "aew", "ahw", "aup", "awb", "axb", "bdl",
  ///     "clb", "eey", "fem", "gka", "jmk", "ksp", "ljm", "lnh", "rms", "rxr", "slp" or "slt"
  /// \param[in] num_workers Number of workers reading audios in parallel.
  /// \param[in] queue_size Connector queue size.
  /// \param[in] data_schema The schema of the CMUArctic dataset.
  /// \param[in] sampler Sampler tells CMUArcticOp what to read.
  CMUArcticOp(const std::string &dataset_dir, const std::string &name, int32_t num_workers, int32_t queue_size,
              std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler);

  /// \brief Destructor.
  ~CMUArcticOp() = default;

  /// \brief A print method typically used for debugging.
  /// \param[out] out The output stream to write output to.
  /// \param[in] show_all A bool to control if you want to show all info or just a summary.
  void Print(std::ostream &out, bool show_all) const override;

  /// \brief Function to count the number of samples in the CMUArctic dataset.
  /// \param[in] dir Path to the CMUArctic directory.
  /// \param[in] name Choose the subset of CMUArctic dataset.
  /// \param[out] count Output arg that will hold the minimum of the actual dataset size and numSamples.
  /// \return Status The status code returned.
  static Status CountTotalRows(const std::string &dir, const std::string &name, int64_t *count);

  /// \brief Op name getter.
  /// \return Name of the current Op.
  std::string Name() const override { return "CMUArcticOp"; }

 private:
  /// \brief Load a tensor row according to a pair.
  /// \param[in] row_id Id for this tensor row.
  /// \param[out] row Audio & label read into this tensor row.
  /// \return Status The status code returned.
  Status LoadTensorRow(row_id_type row_id, TensorRow *row) override;

  /// \brief Parse a single wav file.
  /// \param[in] audio_dir Audio file path.
  /// \param[out] waveform The output waveform tensor.
  /// \return Status The status code returned.
  Status ReadAudio(const std::string &audio_dir, std::shared_ptr<Tensor> *waveform);

  /// \brief Prepare all data in the directory.
  /// \return Status The status code returned.
  Status PrepareData();

  /// \brief Private function for computing the assignment of the column name map.
  /// \return Status.
  Status ComputeColMap() override;

  const std::string name_;
  std::string folder_path_;
  std::string real_path_;
  std::unique_ptr<DataSchema> data_schema_;
  std::vector<std::pair<std::string, std::string>> label_pairs_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_CMU_ARCTIC_OP_H_
