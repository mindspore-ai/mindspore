/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_FOOD101_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_FOOD101_OP_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/engine/datasetops/source/io_block.h"
#include "minddata/dataset/engine/datasetops/source/mappable_leaf_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class Food101Op : public MappableLeafOp {
 public:
  /// \brief Constructor.
  /// \param[in] folder_path Directory of Food101 dataset.
  /// \param[in] usage Usage.
  /// \param[in] num_workers Number of workers reading images in parallel.
  /// \param[in] queue_size Connector queue size.
  /// \param[in] decode Whether to decode images.
  /// \param[in] schema Data schema of Food101 dataset.
  /// \param[in] sampler Sampler tells Food101 what to read.
  Food101Op(const std::string &folder_path, const std::string &usage, int32_t num_workers, int32_t queue_size,
            bool decode, std::unique_ptr<DataSchema> schema, std::shared_ptr<SamplerRT> sampler);

  /// \brief Deconstructor.
  ~Food101Op() override = default;

  /// A print method typically used for debugging.
  /// \param[out] out Out stream.
  /// \param[in] show_all Whether to show all information.
  void Print(std::ostream &out, bool show_all) const override;

  /// Op name getter.
  /// \return Name of the current Op.
  std::string Name() const override { return "Food101Op"; }

  /// Function to count the number of samples in the Food101 dataset.
  /// \param[in] count Output arg that will hold the actual dataset size.
  /// \return The status code returned.
  Status CountTotalRows(int64_t *count);

 private:
  /// Load a tensor row.
  /// \param[in] row_id Row id.
  /// \param[in] row Read all features into this tensor row.
  /// \return The status code returned.
  Status LoadTensorRow(row_id_type row_id, TensorRow *row) override;

  /// \param[in] image_path Path of image data.
  /// \param[in] tensor Get image tensor.
  /// \return The status code returned.
  Status ReadImageToTensor(const std::string &image_path, std::shared_ptr<Tensor> *tensor);

  /// Called first when function is called. Get file_name, img_path info from ".txt" files.
  /// \return Status - The status code returned.
  Status PrepareData();

  /// Private function for computing the assignment of the column name map.
  /// \return Status-the status code returned.
  Status ComputeColMap() override;

  /// Private function for getting all the image files;
  /// \param[in] file_path The path of the dataset.
  /// \return Status-the status code returned.
  Status GetAllImageList(const std::string &file_path);

  std::string folder_path_;  // directory of Food101 folder.
  std::string usage_;
  bool decode_;
  std::unique_ptr<DataSchema> data_schema_;
  std::set<std::string> classes_;
  std::vector<std::string> all_img_lists_;
  std::map<std::string, int32_t> class_index_;
  std::map<std::string, std::vector<int32_t>> annotation_map_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_FOOD101_OP_H_
