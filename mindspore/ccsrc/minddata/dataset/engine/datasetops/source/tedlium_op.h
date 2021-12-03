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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_TEDLIUM_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_TEDLIUM_OP_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/engine/datasetops/source/mappable_leaf_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/engine/ir/cache/dataset_cache.h"

#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/path.h"

namespace mindspore {
namespace dataset {
class TedliumOp : public MappableLeafOp {
 public:
  /// \brief Constructor.
  /// \param[in] dataset_dir Directory of tedlium dataset.
  /// \param[in] release Release of tedlium dataset, can be 'release1', 'release2' or 'release3'.
  /// \param[in] usage Usage of this dataset, if release is release3, can be '', else 'train', 'dev', 'test' or 'all'.
  /// \param[in] extensions Extensions of the sph file, only '.sph' is valid.
  /// \param[in] num_parallel_workers Number of workers in parallel.
  /// \param[in] data_schema Schema of dataset.
  /// \param[in] sampler Sampler tells TedliumOp what to read.
  /// \param[in] queue_size Connector queue size.
  TedliumOp(const std::string &dataset_dir, const std::string &release, const std::string &usage,
            const std::string &extensions, int32_t num_parallel_workers, std::unique_ptr<DataSchema> data_schema,
            std::shared_ptr<SamplerRT> sampler, int32_t queue_size);

  /// \brief Destructor.
  ~TedliumOp() = default;

  /// \brief A print method typically used for debugging.
  /// \param[in] out Out stream.
  /// \param[in] show_all Whether to show all information.
  void Print(std::ostream &out, bool show_all) const override;

  /// \brief Op name getter.
  std::string Name() const override { return "TedliumOp"; }

  /// \brief Initialize TedliumOp related var, calls the function to walk all files.
  /// \return Status The status code returned.
  Status PrepareData() override;

  /// \brief Function to count the number of samples in the TEDLIUM dataset.
  /// \param[in] dataset_dir Directory of tedlium dataset.
  /// \param[in] release Release of tedlium dataset.
  /// \param[in] usage Usage of this dataset, if release is release3, can be '', else 'train', 'dev', 'test' or 'all'.
  /// \param[in] extensions Extensions of the sph file, only '.sph' is valid.
  /// \param[in] count Output arg that will hold the actual dataset size.
  /// \return Status The status code returned.
  static Status CountTotalRows(const std::string &dataset_dir, const std::string &release, const std::string &usage,
                               const std::string &extensions, int64_t *count);

 private:
  /// \brief Read stm file.
  /// \param[in] file_stm_path The path of stm file.
  /// \param[in] row_line Which line of the file we need to read.
  /// \param[out] talk_id Talk identifier of the row_line in the file.
  /// \param[out] speaker_id Speaker identifier of the row_line in the file.
  /// \param[out] start_time Start time of the row_line in the file.
  /// \param[out] end_time End time of the row_line in the file.
  /// \param[out] identifier Identifier of the row_line in the file.
  /// \param[out] transcript Transcript of the row_line in the file.
  /// \return Status The status code returned.
  Status ReadStm(const Path &file_stm_path, int32_t row_line, std::string *talk_id, std::string *speaker_id,
                 std::string *start_time, std::string *end_time, std::string *identifier, std::string *transcript);

  /// \brief Read sph file.
  /// \param[in] file_sph_path The path of sph file.
  /// \param[in] start_time The start_time of row we need to use.
  /// \param[in] end_time The end_time of row we need to use.
  /// \param[out] sample_rate Sample rate of the row.
  /// \param[out] result Waveform result vector of the row.
  /// \return Status The status code returned.
  Status ReadSph(const Path &file_sph_path, double start_time, double end_time, int32_t *sample_rate,
                 std::vector<float> *result);

  /// \brief Read stm files according current release`s usage.
  /// \param[in] stm_folder The folder of stm files.
  /// \param[in] release_usage For release1 or release2, use usage_, for release3, "data".
  /// \return Status The status code returned.
  Status ReadStmFolderRows(const Path &stm_folder, const std::string &release_usage);

  /// \brief Load a tensor row according to a pair.
  /// \param[in] row_id Id of row need to load.
  /// \param[in] row Audio & label read into this tensor row.
  /// \return Status The status code returned.
  Status LoadTensorRow(row_id_type row_id, TensorRow *row) override;

  /// \brief Private function for computing the assignment of the column name map.
  /// \return Status The status code returned.
  Status ComputeColMap() override;

  const std::string release_;
  const std::string dataset_dir_;
  const std::string usage_;
  const std::string extensions_;
  std::unique_ptr<DataSchema> data_schema_;

  std::vector<std::vector<std::string> > audio_files_;
  std::vector<std::string> usage_list_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_TEDLIUM_OP_H_
