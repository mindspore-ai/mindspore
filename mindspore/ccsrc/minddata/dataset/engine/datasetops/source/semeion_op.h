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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SEMEION_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SEMEION_OP_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/engine/datasetops/source/mappable_leaf_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/engine/ir/cache/dataset_cache.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class SemeionOp : public MappableLeafOp {
 public:
  /// \brief Constructor.
  /// \param[in] dataset_dir Directory of semeion dataset.
  /// \param[in] num_parallel_workers Num of workers in parallel.
  /// \param[in] data_schema Schema of dataset.
  /// \param[in] sampler Sampler tells SemeionOp what to read.
  /// \param[in] queue_size Connector queue size.
  SemeionOp(const std::string &dataset_dir, int32_t num_parallel_workers, std::unique_ptr<DataSchema> data_schema,
            std::shared_ptr<SamplerRT> sampler, int32_t queue_size);

  /// \brief Destructor.
  ~SemeionOp() = default;

  /// \brief A print method typically used for debugging.
  /// \param[in] out Out stream.
  /// \param[in] show_all Whether to show all information.
  void Print(std::ostream &out, bool show_all) const override;

  /// \brief Op name getter.
  /// \return Name of the current Op.
  std::string Name() const override { return "SemeionOp"; }

  /// \brief Count total rows.
  /// \param[in] dataset_dir File path.
  /// \param[out] count Get total row.
  /// \return Status The status code returned.
  static Status CountTotalRows(const std::string &dataset_dir, int64_t *count);

  /// \brief Function to count the number of samples in the SemeionOp.
  /// \return Status The status code returned.
  Status PrepareData() override;

 private:
  /// \brief Load a tensor row according to a pair.
  /// \param[in] index Index need to load.
  /// \param[out] trow Image & label read into this tensor row.
  /// \return Status The status code returned.
  Status LoadTensorRow(row_id_type index, TensorRow *trow) override;

  /// \brief Get the img and label according the row_id.
  /// \param[in] index Index of row need to load.
  /// \param[out] img_tensor The image data.
  /// \param[out] label_tensor The label data.
  /// \return Status The status code returned.
  Status TransRowIdResult(row_id_type index, std::shared_ptr<Tensor> *img_tensor,
                          std::shared_ptr<Tensor> *label_tensor);

  /// \brief Private function for computing the assignment of the column name map.
  /// \return Status The status code returned.
  Status ComputeColMap() override;

  const std::string dataset_dir_;
  std::unique_ptr<DataSchema> data_schema_;
  std::vector<std::string> semeionline_rows_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SEMEION_OP_H_
