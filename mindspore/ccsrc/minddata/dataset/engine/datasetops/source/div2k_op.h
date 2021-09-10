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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_DIV2K_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_DIV2K_OP_H_

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
class DIV2KOp : public MappableLeafOp {
 public:
  /// \brief Constructor.
  /// \param[in] int32_t num_workers - num of workers reading images in parallel.
  /// \param[in] std::string dataset_dir - dir directory of DIV2K dataset.
  /// \param[in] std::string usage - the type of dataset. Acceptable usages include "train", "valid" or "all".
  /// \param[in] std::string downgrade - the mode of downgrade. Acceptable downgrades include "bicubic", "unknown",
  ///    "mild", "difficult" or "wild".
  /// \param[in] int32_t scale - the scale of downgrade. Acceptable scales include 2, 3, 4 or 8.
  /// \param[in] bool decode - decode the images after reading.
  /// \param[in] int32_t queue_size - connector queue size.
  /// \param[in] DataSchema data_schema - the schema of each column in output data.
  /// \param[in] std::unique_ptr<Sampler> sampler - sampler tells ImageFolderOp what to read.
  DIV2KOp(int32_t num_workers, const std::string &dataset_dir, const std::string &usage, const std::string &downgrade,
          int32_t scale, bool decode, int32_t queue_size, std::unique_ptr<DataSchema> data_schema,
          std::shared_ptr<SamplerRT> sampler);

  /// \brief Destructor.
  ~DIV2KOp() = default;

  /// \brief A print method typically used for debugging.
  /// \param[out] out
  /// \param[in] show_all
  void Print(std::ostream &out, bool show_all) const override;

  /// \brief Function to count the number of samples in the DIV2K dataset.
  /// \param[in] dir - path to the DIV2K directory.
  /// \param[in] usage - the type of dataset.  Acceptable usages include "train", "valid" or "all".
  /// \param[in] downgrade - the mode of downgrade. Acceptable downgrades include "bicubic", "unknown",
  ///    "mild", "difficult" or "wild".
  /// \param[in] scale - the scale of downgrade. Acceptable scales include 2, 3, 4 or 8.
  /// \param[out] count - output arg that will hold the actual dataset size.
  /// \return Status - The status code returned.
  static Status CountTotalRows(const std::string &dir, const std::string &usage, const std::string &downgrade,
                               int32_t scale, int64_t *count);

  /// \brief Op name getter.
  /// \return Name of the current Op.
  std::string Name() const override { return "DIV2KOp"; }

 private:
  /// \brief Load a tensor row according to a pair.
  /// \param[in] uint64_t index - index need to load.
  /// \param[out] TensorRow row - image & label read into this tensor row.
  /// \return Status - The status code returned.
  Status LoadTensorRow(row_id_type index, TensorRow *trow) override;

  /// \brief Called first when function is called.
  /// \return Status - The status code returned.
  Status LaunchThreadsAndInitOp() override;

  /// \brief Get the real name of high resolution images and low resolution images dir in DIV2K dataset.
  /// \param[in] hr_dir_key - the key of high resolution images dir.
  /// \param[in] lr_dir_key - the key of high resolution images dir.
  /// \return Status - The status code returned.
  Status GetDIV2KLRDirRealName(const std::string &hr_dir_key, const std::string &lr_dir_key);

  /// \brief Parse DIV2K data.
  /// \return Status - The status code returned.
  Status ParseDIV2KData();

  /// \brief Get DIV2K data by usage.
  /// \return Status - The status code returned.
  Status GetDIV2KDataByUsage();

  /// \brief Count label index,num rows and num samples.
  /// \return Status - The status code returned.
  Status CountDatasetInfo();

  /// \brief Private function for computing the assignment of the column name map.
  /// \return Status - The status code returned.
  Status ComputeColMap() override;

  std::string dataset_dir_;
  std::string usage_;
  int32_t scale_;
  std::string downgrade_;
  bool decode_;
  std::unique_ptr<DataSchema> data_schema_;

  std::vector<std::pair<std::string, std::string>> image_hr_lr_pairs_;
  std::string hr_dir_real_name_;
  std::string lr_dir_real_name_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_DIV2K_OP_H_
