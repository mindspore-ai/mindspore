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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SUN397_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SUN397_OP_H_

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
/// \brief Forward declares.
template <typename T>
class Queue;

using SUN397LabelPair = std::pair<std::shared_ptr<Tensor>, uint32_t>;

class SUN397Op : public MappableLeafOp {
 public:
  /// \brief Constructor.
  /// \param[in] file_dir Dir directory of SUN397Dataset.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] num_workers Num of workers reading images in parallel.
  /// \param[in] queue_size Connector queue size.
  /// \param[in] data_schema Schema of data.
  /// \param[in] sampler Sampler tells SUN397Op what to read.
  SUN397Op(const std::string &file_dir, bool decode, int32_t num_workers, int32_t queue_size,
           std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler);

  /// \brief Destructor.
  ~SUN397Op() = default;

  /// \brief Method derived from RandomAccess Op, enable Sampler to get all ids for each class.
  /// \param[in] cls_ids Key label, val all ids for this class.
  /// \return The status code returned.
  Status GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const override;

  /// \brief A print method typically used for debugging.
  /// \param[out] out The output stream to write output to.
  /// \param[in] show_all A bool to control if you want to show all info or just a summary.
  void Print(std::ostream &out, bool show_all) const override;

  /// \param[in] dir Path to the PhotoTour directory.
  /// \param[in] decode Decode jpg format images.
  /// \param[out] count Output arg that will hold the minimum of the actual dataset
  ///     size and numSamples.
  /// \return The status code returned.
  static Status CountTotalRows(const std::string &dir, bool decode, int64_t *count);

  /// \brief Op name getter.
  /// \return Name of the current Op.
  std::string Name() const override { return "SUN397Op"; }

 private:
  /// \brief Load a tensor row according to a pair.
  /// \param[in] row_id Id for this tensor row.
  /// \param[out] row Image & label read into this tensor row.
  /// \return Status The status code returned
  Status LoadTensorRow(row_id_type row_id, TensorRow *row);

  /// \brief The content in the given file path.
  /// \param[in] info_file Info file name.
  /// \param[out] ans Store the content of the info file.
  /// \return Status The status code returned
  Status GetFileContent(const std::string &info_file, std::string *ans);

  /// \brief Load the meta information of categories.
  /// \param[in] category_meta_name Category file name.
  /// \return Status The status code returned.
  Status LoadCategories(const std::string &category_meta_name);

  /// \brief Initialize SUN397Op related var, calls the function to walk all files.
  /// \return Status The status code returned.
  Status PrepareData() override;

  /// \brief Private function for computing the assignment of the column name map.
  /// \return Status The status code returned.
  Status ComputeColMap() override;

  int64_t buf_cnt_;
  std::unique_ptr<DataSchema> data_schema_;

  std::string folder_path_;  // directory of image folder
  const bool decode_;
  std::map<std::string, uint32_t> categorie2id_;
  std::vector<std::pair<std::string, uint32_t>> image_path_label_pairs_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SUN397_OP_H_
