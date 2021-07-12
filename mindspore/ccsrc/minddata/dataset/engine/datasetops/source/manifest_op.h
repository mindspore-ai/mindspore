/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_MANIFEST_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_MANIFEST_OP_H_

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
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/util/services.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/wait_post.h"

namespace mindspore {
namespace dataset {
class ManifestOp : public MappableLeafOp {
 public:
  // Constructor
  // @param int32_t num_works - Num of workers reading images in parallel
  // @param std::string - file list of Manifest
  // @param int32_t queue_size - connector queue size
  // @param td::unique_ptr<Sampler> sampler - sampler tells ImageFolderOp what to read
  ManifestOp(int32_t num_works, std::string file, int32_t queue_size, bool decode,
             const std::map<std::string, int32_t> &class_index, std::unique_ptr<DataSchema> data_schema,
             std::shared_ptr<SamplerRT> sampler, std::string usage);
  // Destructor.
  ~ManifestOp() = default;

  // Method derived from RandomAccess Op, enable Sampler to get all ids for each class
  // @param (std::map<int64_t, std::vector<int64_t >> * map - key label, val all ids for this class
  // @return Status The status code returned
  Status GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const override;

  // A print method typically used for debugging
  // @param out
  // @param show_all
  void Print(std::ostream &out, bool show_all) const override;

  /// \brief Counts the total number of rows in Manifest
  /// \param[out] count Number of rows counted
  /// \return Status of the function
  Status CountTotalRows(int64_t *count);

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return "ManifestOp"; }

  /// \brief Base-class override for GetNumClasses
  /// \param[out] num_classes the number of classes
  /// \return Status of the function
  Status GetNumClasses(int64_t *num_classes) override;

  /// \brief Gets the class indexing
  /// \return Status - The status code return
  Status GetClassIndexing(std::vector<std::pair<std::string, std::vector<int32_t>>> *output_class_indexing) override;

 private:
  // Load a tensor row according to a pair
  // @param row_id_type row_id - id for this tensor row
  // @param std::pair<std::string, std::vector<std::string>> - <imagefile, <label1, label2...>>
  // @param TensorRow row - image & label read into this tensor row
  // @return Status The status code returned
  Status LoadTensorRow(row_id_type row_id, TensorRow *row) override;

  // Parse manifest file to get image path and label and so on.
  // @return Status The status code returned
  Status ParseManifestFile();

  // Called first when function is called
  // @return Status The status code returned
  Status LaunchThreadsAndInitOp() override;

  // Check if image ia valid.Only support JPEG/PNG/GIF/BMP
  // @return
  Status CheckImageType(const std::string &file_name, bool *valid);

  // Count label index,num rows and num samples
  // @return Status The status code returned
  Status CountDatasetInfo();

  // Private function for computing the assignment of the column name map.
  // @return - Status
  Status ComputeColMap() override;

  int64_t io_block_pushed_;
  int64_t sampler_ind_;
  std::unique_ptr<DataSchema> data_schema_;
  std::string file_;  // file that store the information of images
  std::map<std::string, int32_t> class_index_;
  bool decode_;
  std::string usage_;

  std::map<std::string, int32_t> label_index_;
  std::vector<std::pair<std::string, std::vector<std::string>>> image_labelname_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_MANIFEST_OP_H_
