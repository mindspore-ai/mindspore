/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_CALTECH_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_CALTECH_OP_H_

#include <memory>
#include <string>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/engine/datasetops/source/image_folder_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"

namespace mindspore {
namespace dataset {
/// \brief Read Caltech256 Dataset.
class CaltechOp : public ImageFolderOp {
 public:
  /// \brief Constructor.
  /// \param[in] num_workers Num of workers reading images in parallel.
  /// \param[in] file_dir Directory of caltech dataset.
  /// \param[in] queue_size Connector queue size.
  /// \param[in] do_decode Whether to decode the raw data.
  /// \param[in] data_schema Data schema of caltech256 dataset.
  /// \param[in] sampler Sampler tells CaltechOp what to read.
  CaltechOp(int32_t num_workers, const std::string &file_dir, int32_t queue_size, bool do_decode,
            std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler);

  /// \brief Destructor.
  ~CaltechOp() = default;

  /// \brief Op name getter.
  /// \return Name of the current Op.
  std::string Name() const override { return "CaltechOp"; }

  /// \brief DatasetName name getter.
  /// \param[in] upper Whether the returned name begins with uppercase.
  /// \return DatasetName of the current Op.
  std::string DatasetName(bool upper = false) const { return upper ? "Caltech" : "caltech"; }
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_CALTECH_OP_H_
