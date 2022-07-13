/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_QMNIST_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_QMNIST_OP_H_

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
#include "minddata/dataset/engine/datasetops/source/mnist_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/wait_post.h"

namespace mindspore {
namespace dataset {

using QMnistImageInfoPair = std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>;

class QMnistOp : public MnistOp {
 public:
  // Constructor.
  // @param const std::string &folder_path - dir directory of QMNIST data file.
  // @param const std::string &usage - Usage of this dataset, can be 'train', 'test', 'test10k', 'test50k', 'nist' or
  //     'all'.
  // @param bool compat - Compatibility with Mnist.
  // @param std::unique_ptr<DataSchema> data_schema - the schema of the QMNIST dataset.
  // @param td::unique_ptr<Sampler> sampler - sampler tells QMnistOp what to read.
  // @param int32_t num_workers - number of workers reading images in parallel.
  // @param int32_t queue_size - connector queue size.
  QMnistOp(const std::string &folder_path, const std::string &usage, bool compat,
           std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler, int32_t num_workers,
           int32_t queue_size);

  // Destructor.
  ~QMnistOp() = default;

  // Op name getter.
  // @return std::string - Name of the current Op.
  std::string Name() const override { return "QMnistOp"; }

  // DatasetName name getter
  // \return std::string - DatasetName of the current Op
  std::string DatasetName(bool upper = false) const { return upper ? "QMnist" : "qmnist"; }

  // A print method typically used for debugging.
  // @param std::ostream &out - out stream.
  // @param bool show_all - whether to show all information.
  void Print(std::ostream &out, bool show_all) const override;

  // Function to count the number of samples in the QMNIST dataset.
  // @param const std::string &dir - path to the QMNIST directory.
  // @param const std::string &usage - Usage of this dataset, can be 'train', 'test', 'test10k', 'test50k', 'nist' or
  //     'all'.
  // @param int64_t *count - output arg that will hold the actual dataset size.
  // @return Status -The status code returned.
  static Status CountTotalRows(const std::string &dir, const std::string &usage, int64_t *count);

 private:
  // Load a tensor row according to a pair.
  // @param row_id_type row_id - id for this tensor row.
  // @param TensorRow row - image & label read into this tensor row.
  // @return Status - The status code returned.
  Status LoadTensorRow(row_id_type row_id, TensorRow *trow) override;

  // Get needed files in the folder_path_.
  // @return Status - The status code returned.
  Status WalkAllFiles() override;

  // Read images and labels from the file stream.
  // @param std::ifstream *image_reader - image file stream.
  // @param std::ifstream *label_reader - label file stream.
  // @param size_t index - the index of file that is reading.
  // @return Status The status code returned.
  Status ReadImageAndLabel(std::ifstream *image_reader, std::ifstream *label_reader, size_t index) override;

  // Check label stream.
  // @param const std::string &file_name - label file name.
  // @param std::ifstream *label_reader - label file stream.
  // @param uint32_t num_labels - returns the number of labels.
  // @return Status The status code returned.
  Status CheckLabel(const std::string &file_name, std::ifstream *label_reader, uint32_t *num_labels) override;

  const bool compat_;  // compatible with mnist

  std::vector<QMnistImageInfoPair> image_info_pairs_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_QMNIST_OP_H_
