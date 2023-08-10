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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_WIDER_FACE_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_WIDER_FACE_OP_H_

#include <map>
#include <memory>
#include <string>
#include <set>
#include <utility>
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
class WIDERFaceOp : public MappableLeafOp {
 public:
  /// Constructor.
  /// \param[in] string folder_path - directory of WIDERFace dataset.
  /// \param[in] string usage - usage.
  /// \param[in] uint32_t num_workers - number of workers reading images in parallel.
  /// \param[in] uint32_t queue_size - connector queue size.
  /// \param[in] bool decode - whether to decode images.
  /// \param[in] unique_ptr<DataSchema> schema - data schema of WIDERFace dataset.
  /// \param[in] shared_ptr<Sampler> sampler - sampler tells WIDERFace what to read.
  WIDERFaceOp(const std::string &folder_path, const std::string &usage, int32_t num_workers, int32_t queue_size,
              bool decode, std::unique_ptr<DataSchema> schema, std::shared_ptr<SamplerRT> sampler);

  /// Deconstructor.
  ~WIDERFaceOp() override = default;

  /// A print method typically used for debugging.
  /// \param[out] out - out stream.
  /// \param[in] show_all - whether to show all information.
  void Print(std::ostream &out, bool show_all) const override;

  /// Op name getter.
  /// \return Name of the current Op.
  std::string Name() const override { return "WIDERFaceOp"; }

  /// Function to count the number of samples in the WIDERFace dataset.
  /// \param[in] count - output arg that will hold the actual dataset size.
  /// \return Status - The status code returned.
  Status CountTotalRows(int64_t *count);

 private:
  /// Load a tensor row.
  /// \param[in] uint64_t row_id - row id.
  /// \param[in] TensorRow row - read all features into this tensor row.
  /// \return Status - The status code returned.
  Status LoadTensorRow(row_id_type row_id, TensorRow *row) override;

  /// \param[in] string image_path - path of image data.
  /// \param[in] Tensor &tensor - get image tensor.
  /// \return Status - The status code returned.
  Status ReadImageToTensor(const std::string &image_path, std::shared_ptr<Tensor> *tensor);

  /// Called first when function is called. Get file_name, img_path and attribute info from ".txt" files.
  /// \return Status - The status code returned.
  Status PrepareData();

  /// \param[in] string wf_path - walk the selected folder to get image names.
  /// \return Status - The status code returned.
  Status WalkFolders(const std::string &wf_path);

  /// Get all valid or train or both file paths(names).
  /// \param[in] string list_path - real path of annotation file.
  /// \param[in] string image_folder_path - real path of image folder.
  /// \return Status - the status code returned.
  Status GetTraValAnno(const std::string &list_path, const std::string &image_folder_path);

  /// \param[in] string path - path to the WIDERFace directory.
  /// \param[in] TensorRow tensor - get attributes.
  /// \return Status-the status code returned.
  Status ParseAnnotations(const std::string &path, TensorRow *tensor);

  /// Split attribute info with space.
  /// \param[in] string line - read line from annotation files.
  /// \param[out] vector split_num - vector of annotation values.
  /// \return Status-the status code returned.
  Status Split(const std::string &line, std::vector<int32_t> *split_num);

  /// Private function for computing the assignment of the column name map.
  /// \return Status-the status code returned.
  Status ComputeColMap() override;

  std::string folder_path_;  // directory of WIDERFace folder.
  std::string usage_;
  bool decode_;
  std::unique_ptr<DataSchema> data_schema_;

  std::vector<std::string> all_img_names_;
  std::set<std::string> folder_names_;
  std::map<std::string, int32_t> class_index_;
  std::map<std::string, std::vector<int32_t>> annotation_map_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_WIDER_FACE_OP_H_
