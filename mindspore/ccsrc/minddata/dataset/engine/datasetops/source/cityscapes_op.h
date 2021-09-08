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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_CITYSCAPES_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_CITYSCAPES_OP_H_

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
class CityscapesOp : public MappableLeafOp {
 public:
  /// \brief Constructor.
  /// \param[in] int32_t num_workers - num of workers reading images in parallel.
  /// \param[in] std::string dataset_dir - dir directory of Cityscapes dataset.
  /// \param[in] std::string usage - the type of dataset. Acceptable usages include "train", "test", "val" or "all" if
  ///     quality_mode is "fine" otherwise "train", "train_extra", "val" or "all".
  /// \param[in] std::string quality_mode - the quality mode of processed image. Acceptable quality_modes include
  ///     "fine" or "coarse".
  /// \param[in] std::string task - the type of task which is used to select output data. Acceptable tasks include
  ///     "instance", "semantic", "polygon" or "color".
  /// \param[in] bool decode - decode the images after reading.
  /// \param[in] int32_t queue_size - connector queue size.
  /// \param[in] DataSchema data_schema - the schema of each column in output data.
  /// \param[in] std::unique_ptr<Sampler> sampler - sampler tells ImageFolderOp what to read.
  CityscapesOp(int32_t num_workers, const std::string &dataset_dir, const std::string &usage,
               const std::string &quality_mode, const std::string &task, bool decode, int32_t queue_size,
               std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler);

  /// \brief Destructor.
  ~CityscapesOp() = default;

  /// \brief A print method typically used for debugging.
  /// \param[out] out.
  /// \param[in] show_all.
  void Print(std::ostream &out, bool show_all) const override;

  /// \brief Function to count the number of samples in the Cityscapes dataset.
  /// \param[in] dir - path to the Cityscapes directory.
  /// \param[in] usage - the type of dataset. Acceptable usages include "train", "test", "val" or "all" if
  ///     quality_mode is "fine" otherwise "train", "train_extra", "val" or "all".
  /// \param[in] quality_mode - the quality mode of processed image. Acceptable quality_modes include
  ///     "fine" or "coarse".
  /// \param[in] task - the type of task which is used to select output data. Acceptable tasks include
  ///     "instance", "semantic", "polygon" or "color".
  /// \param[out] count - output arg that will hold the actual dataset size.
  /// \return Status - The status code returned.
  static Status CountTotalRows(const std::string &dir, const std::string &usage, const std::string &quality_mode,
                               const std::string &task, int64_t *count);

  /// \brief Op name getter.
  /// \return Name of the current Op.
  std::string Name() const override { return "CityscapesOp"; }

 private:
  /// \brief Load a tensor row according to a pair.
  /// \param[in] uint64_t index - index need to load.
  /// \param[out] TensorRow row - image & task read into this tensor row.
  /// \return Status - The status code returned.
  Status LoadTensorRow(row_id_type index, TensorRow *trow) override;

  /// \brief Called first when function is called.
  /// \return Status - The status code returned.
  Status LaunchThreadsAndInitOp() override;

  /// \brief Parse Cityscapes data.
  /// \return Status - The status code returned.
  Status ParseCityscapesData();

  /// \brief Get Cityscapes data by usage.
  /// \param[in] images_dir - path to the images in the dataset.
  /// \param[in] task_dir - path to the given task file.
  /// \param[in] real_quality_mode - the real quality mode of image in dataset.
  /// \return Status - The status code returned.
  Status GetCityscapesDataByUsage(const std::string &images_dir, const std::string &task_dir,
                                  const std::string &real_quality_mode);

  /// \brief Count label index, num rows and num samples.
  /// \return Status - The status code returned.
  Status CountDatasetInfo();

  /// \brief Private function for computing the assignment of the column name map.
  /// \return Status - The status code returned.
  Status ComputeColMap() override;

  /// \brief Private function for get the task suffix.
  /// \param[in] task - the type of task which is used to select output data.
  /// \param[in] real_quality_mode - the real quality mode of image in dataset.
  /// \return std::string - the suffix of task file.
  std::string GetTaskSuffix(const std::string &task, const std::string &real_quality_mode);

  std::string dataset_dir_;
  std::string usage_;
  std::string quality_mode_;
  std::string task_;
  bool decode_;
  std::unique_ptr<DataSchema> data_schema_;

  std::vector<std::pair<std::string, std::string>> image_task_pairs_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_CITYSCAPES_OP_H_
