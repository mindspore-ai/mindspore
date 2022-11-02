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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_PLACES365_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_PLACES365_OP_H_

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
// Forward declares
template <typename T>
class Queue;

using Places365LabelPair = std::pair<std::shared_ptr<Tensor>, uint32_t>;

class Places365Op : public MappableLeafOp {
 public:
  // Constructor.
  // @param const std::string &root - Path to the root directory that contains the dataset.
  // @param const std::string &usage - Usage of this dataset, can be
  //     'train-standard', 'train-challenge' or 'val'.
  // @param bool small - Use high resolution images or 256*256 resolution images.
  // @param bool decode - Decode  jpg format images.
  // @param int32_t num_workers - number of workers reading images in parallel.
  // @param int32_t queue_size - connector queue size.
  // @param std::unique_ptr<DataSchema> data_schema - the schema of the places365 dataset.
  // @param std::unique_ptr<Sampler> sampler - sampler tells Places365Op what to read.
  Places365Op(const std::string &root, const std::string &usage, bool small, bool decode, int32_t num_workers,
              int32_t queue_size, std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler);

  // Destructor.
  ~Places365Op() = default;

  // Method derived from RandomAccess Op, enable Sampler to get all ids for each class.
  // @param std::map<int32_t, std::vector<int64_t >> *cls_ids - key label, val all ids for this class.
  // @return Status - The status code returned.
  Status GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const override;

  // A print method typically used for debugging.
  // @param std::ostream &out - out stream.
  // @param bool show_all - whether to show all information.
  void Print(std::ostream &out, bool show_all) const override;

  // Function to count the number of samples in the PhotoTour dataset.
  // @param const std::string &dir - path to the PhotoTour directory.
  // @param const std::string &usage - Usage of this dataset, can be
  //     'train-standard', 'train-challenge' or 'val'.
  // @param bool small - Use high resolution images or 256*256 resolution images.
  // @param bool decode - Decode  jpg format images.
  // @param int64_t *count - output arg that will hold the minimum of the actual dataset
  //     size and numSamples.
  // @return Status - The status code returned.
  static Status CountTotalRows(const std::string &dir, const std::string &usage, const bool small, const bool decode,
                               int64_t *count);

  // Op name getter.
  // @return std::string - Name of the current Op.
  std::string Name() const override { return "Places365Op"; }

 private:
  // Load a tensor row according to the row_id.
  // @param row_id_type row_id - id for this tensor row.
  // @param TensorRow *row - load one piece of data into this tensor row.
  // @return Status - The status code returned.
  Status LoadTensorRow(row_id_type row_id, TensorRow *row);

  // Read the content in the given file path.
  // @param const std::string &info_file - info file name.
  // @param std::string *ans - store the content of the info file.
  // @return Status - The status code returned.
  Status GetFileContent(const std::string &info_file, std::string *ans);

  // Load the meta information of categories.
  // @param const std::string &category_meta_name - category file name.
  // @return Status - The status code returned.
  Status LoadCategories(const std::string &category_meta_name);

  // Load the meta information of file information.
  // @param const std::string &filelists_meta_name - meta file name.
  // @return Status - The status code returned.
  Status LoadFileLists(const std::string &filelists_meta_name);

  // Get one piece of places365 data.
  // @param uint32_t index Index of the data.
  // @param std::shared_ptr<Tensor> *image_tensor - Store the result in image_tensor.
  // @return Status - The status code returned.
  Status GetPlaces365DataTensor(uint32_t index, std::shared_ptr<Tensor> *image_tensor);

  // Parse all places365 dataset files.
  // @return Status The status code returned.
  Status PrepareData() override;

  // Private function for computing the assignment of the column name map.
  // @return Status - The status code returned.
  Status ComputeColMap() override;

  int64_t buf_cnt_;
  std::unique_ptr<DataSchema> data_schema_;

  const std::string root_;   // directory of image folder
  const std::string usage_;  // can only be "train-challenge", "train-standard" or "val"
  const bool small_;
  const bool decode_;
  std::map<std::string, int> categorie2id_;
  std::vector<std::pair<std::string, uint32_t>> image_path_label_pairs_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_PLACES365_OP_H_
