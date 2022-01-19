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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_GTZAN_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_GTZAN_OP_H_

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
class GTZANOp : public MappableLeafOp {
 public:
  /// \brief Constructor
  /// \param[in] usage Usage of this dataset, can be 'train', 'valid', 'test', or 'all'.
  /// \param[in] num_workers Number of workers reading audios in parallel.
  /// \param[in] folder_path Dir directory of GTZAN.
  /// \param[in] queue_size Connector queue size.
  /// \param[in] data_schema The schema of the GTZAN dataset.
  /// \param[in] sampler Sampler tells GTZANOp what to read.
  GTZANOp(const std::string &usage, int32_t num_workers, const std::string &folder_path, int32_t queue_size,
          std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler);

  /// \Destructor.
  ~GTZANOp() = default;

  /// \A print method typically used for debugging.
  /// \param[out] out Output stream.
  /// \param[in] show_all Whether to show all information.
  void Print(std::ostream &out, bool show_all) const override;

  /// \Function to count the number of samples in the GTZAN dataset.
  /// \param[in] dir Path to the GTZAN directory.
  /// \param[in] usage Choose the subset of GTZAN dataset.
  /// \param[out] count Output arg that will hold the actual dataset size.
  /// \return Status The status code returned.
  static Status CountTotalRows(const std::string &dir, const std::string &usage, int64_t *count);

  /// \Op name getter.
  /// \return Name of the current Op.
  std::string Name() const override { return "GTZANOp"; }

 private:
  /// \Load a tensor row according to a pair.
  /// \param[in] row_id Id for this tensor row.
  /// \param[out] row Audio & label read into this tensor row.
  /// \return Status The status code returned.
  Status LoadTensorRow(row_id_type row_id, TensorRow *row) override;

  /// \Parse a audio file.
  /// \param[in] audio_dir Audio file path.
  /// \param[out] waveform The output waveform tensor.
  /// \return Status The status code returned.
  Status ReadAudio(const std::string &audio_dir, std::shared_ptr<Tensor> *waveform);

  /// \Prepare data.
  /// \return Status The status code returned.
  Status PrepareData();

  /// \Private function for computing the assignment of the column name map.
  /// \return Status The status code returned.
  Status ComputeColMap() override;

  const std::string usage_;
  std::string folder_path_;
  std::unique_ptr<DataSchema> data_schema_;
  std::vector<std::pair<std::string, std::string>> audio_names_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_GTZAN_OP_H_
