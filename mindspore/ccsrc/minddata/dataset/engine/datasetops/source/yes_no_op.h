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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_YES_NO_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_YES_NO_OP_H_

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
#include "minddata/dataset/util/services.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/wait_post.h"

namespace mindspore {
namespace dataset {
class YesNoOp : public MappableLeafOp {
 public:
  /// Constructor.
  /// @param std::string file_dir - dir directory of YesNo.
  /// @param int32_t num_workers - number of workers reading images in parallel.
  /// @param int32_t queue_size - connector queue size.
  /// @param std::unique_ptr<DataSchema> data_schema - the schema of the YesNo dataset.
  /// @param std::shared_ptr<Sampler> sampler - sampler tells YesNoOp what to read.
  YesNoOp(const std::string &file_dir, int32_t num_workers, int32_t queue_size, std::unique_ptr<DataSchema> data_schema,
          std::shared_ptr<SamplerRT> sampler);

  /// Destructor.
  ~YesNoOp() = default;

  /// A print method typically used for debugging.
  /// @param std::ostream &out - out stream.
  /// @param bool show_all - whether to show all information.
  void Print(std::ostream &out, bool show_all) const override;

  /// Op name getter.
  /// @return Name of the current Op.
  std::string Name() const override { return "YesNoOp"; }

  /// @param int64_t *count - output rows number of YesNoDataset.
  /// @return Status - The status code returned.
  Status CountTotalRows(int64_t *count);

 private:
  /// Load a tensor row according to wave id.
  /// @param row_id_type row_id - id for this tensor row.
  /// @param TensorRow trow - wave & target read into this tensor row.
  /// @return Status - The status code returned.
  Status LoadTensorRow(row_id_type row_id, TensorRow *trow) override;

  /// Get file infos by file name.
  /// @param string line - file name.
  /// @param vector split_num - vector of annotation.
  /// @return Status - The status code returned.
  Status Split(const std::string &line, std::vector<int32_t> *split_num);

  /// Initialize YesNoDataset related var, calls the function to walk all files.
  /// @return Status - The status code returned.
  Status PrepareData();

  /// Private function for computing the assignment of the column name map.
  /// @return Status - The status code returned.
  Status ComputeColMap() override;

  std::vector<std::string> all_wave_files_;
  std::string dataset_dir_;
  std::unique_ptr<DataSchema> data_schema_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_YES_NO_OP_H
