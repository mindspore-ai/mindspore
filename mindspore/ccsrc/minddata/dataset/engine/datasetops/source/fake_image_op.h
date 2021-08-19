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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_FAKE_IMAGE_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_FAKE_IMAGE_OP_H_

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

class FakeImageOp : public MappableLeafOp {
 public:
  // Constructor.
  // @param int32_t num_images - Number of generated fake images.
  // @param const std::vector<int32_t> &image_size - The size of fake image.
  // @param int32_t num_classes - Number of classes in fake images.
  // @param int32_t base_seed - A base seed which is used in generating fake image randomly.
  // @param int32_t num_workers - Number of workers reading images in parallel.
  // @param int32_t op_connector_size - Connector queue size.
  // @param std::unique_ptr<DataSchema> data_schema - The schema of the fake image dataset.
  // @param td::unique_ptr<Sampler> sampler - Sampler tells FakeImageOp what to read.
  FakeImageOp(int32_t num_images, const std::vector<int32_t> &image_size, int32_t num_classes, int32_t base_seed,
              int32_t num_workers, int32_t op_connector_size, std::unique_ptr<DataSchema> data_schema,
              std::shared_ptr<SamplerRT> sampler);

  // Destructor.
  ~FakeImageOp() = default;

  // Method derived from RandomAccess Op, enable Sampler to get all ids for each class.
  // @param std::map<int32_t, std::vector<int64_t>> *cls_ids - Key label, val all ids for this class.
  // @return Status The status code returned.
  Status GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const override;

  // A print method typically used for debugging.
  // @param out - The output stream to write output to.
  // @param show_all - A bool to control if you want to show all info or just a summary.
  void Print(std::ostream &out, bool show_all) const override;

  // Function to count the number of samples in the FakeImage dataset.
  // @return Number of images.
  int64_t GetTotalRows() const { return num_images_; }

  // Op name getter.
  // @return Name of the current Op.
  std::string Name() const override { return "FakeImageOp"; }

  // Get a image from index
  // @param int32_t index - Generate one image according to index.
  Status GetItem(int32_t index);

 private:
  // Load a tensor row according to a lable_list.
  // @param row_id_type row_id - Id for this tensor row.
  // @param TensorRow *row - Image & label read into this tensor row.
  // @return Status The status code returned.
  Status LoadTensorRow(row_id_type row_id, TensorRow *row) override;

  // Generate all labels of FakeImage dataset
  // @return Status The status code returned.
  Status PrepareData();

  // Private function for computing the assignment of the column name map.
  // @return Status The status code returned.
  Status ComputeColMap() override;

  int32_t num_images_;
  int32_t base_seed_;
  std::vector<int> image_size_;
  int32_t num_classes_;

  int64_t rows_per_buffer_;
  std::unique_ptr<DataSchema> data_schema_;

  int32_t image_total_size_;
  std::vector<uint32_t> label_list_;
  std::vector<std::shared_ptr<Tensor>> image_tensor_;
  std::mt19937 rand_gen_;
  std::mutex access_mutex_;
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_FAKE_IMAGE_OP_H_
