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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_STL10_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_STL10_OP_H_

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
class STL10Op : public MappableLeafOp {
 public:
  // Constructor
  // @param const std::string &usage - Usage of this dataset, can be 'train', 'test', 'unlabeled', "train+unlabeled" or
  //     "all"
  // @param int32_t num_workers - number of workers reading images in parallel.
  // @param std::string folder_path - dir directory of stl10.
  // @param int32_t queue_size - connector queue size.
  // @param std::unique_ptr<DataSchema> data_schema - the schema of the stl10 dataset.
  // @param td::unique_ptr<Sampler> sampler - sampler tells STL10Op what to read.
  STL10Op(const std::string &usage, int32_t num_workers, const std::string &folder_path, int32_t queue_size,
          std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler);

  // Destructor
  ~STL10Op() = default;

  // Method derived from RandomAccess Op, enable Sampler to get all ids for each class
  // @param (std::map<int32_t, std::vector<int64_t>> * cls_ids - key label, val all ids for this class
  // @return Status The status code returned
  Status GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const override;

  // A print method typically used for debugging.
  // @param out - The output stream to write output to.
  // @param show_all - A bool to control if you want to show all info or just a summary.
  void Print(std::ostream &out, bool show_all) const override;

  // Function to count the number of samples in the STL10 dataset.
  // @param dir path to the STL10 directory.
  // @param const std::string &usage - Usage of this dataset, can be 'train', 'test', 'unlabeled', "train+unlabeled" or
  //     "all"
  // @param count output arg that will hold the minimum of the actual dataset size and numSamples.
  // @return Status The status code returned.
  static Status CountTotalRows(const std::string &dir, const std::string &usage, int64_t *count);

  // Op name getter.
  // @return Name of the current Op.
  std::string Name() const override { return "STL10Op"; }

 private:
  // Read the needed files in the directory, save the file in image_names_ and label_names_.
  // @return Status The status code returned.
  Status WalkAllFiles();

  // Read the specified number of images and labels from the file stream.
  // @param std::ifstream *image_reader - image file stream.
  // @param std::ifstream *label_reader - label file stream.
  // @param int64_t index - number of image to read.
  // @return Status The status code returned.
  Status ReadImageAndLabel(std::ifstream *image_reader, std::ifstream *label_reader, size_t index);

  // Parse all stl10 dataset files
  // @return Status The status code returned
  Status ParseSTLData();

  // Read all stl10 data files in the directory
  // @return Status The status code returned.
  Status PrepareData() override;

  // Private function for computing the assignment of the column name map.
  // @return - Status.
  Status ComputeColMap() override;

  // Load a tensor row according to a pair.
  // @param uint64_t index - index need to load.
  // @param TensorRow trow - image & label read into this tensor row.
  // @return Status The status code returned.
  Status LoadTensorRow(row_id_type index, TensorRow *trow) override;

  std::string folder_path_;  // directory of image folder.
  const std::string usage_;  // can only be either "train" or "test" or "unlabeled" or "train+unlabeled" or "all".
  std::unique_ptr<DataSchema> data_schema_;

  std::vector<std::pair<std::shared_ptr<Tensor>, int32_t>> stl10_image_label_pairs_;
  std::vector<std::string> image_names_;
  std::vector<std::string> label_names_;
  std::vector<std::string> image_path_;
  std::vector<std::string> label_path_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_STL10_OP_H_
