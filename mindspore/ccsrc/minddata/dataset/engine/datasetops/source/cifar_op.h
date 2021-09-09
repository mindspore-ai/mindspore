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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_CIFAR_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_CIFAR_OP_H_

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
class CifarOp : public MappableLeafOp {
 public:
  enum CifarType { kCifar10, kCifar100 };

  // Constructor
  // @param CifarType type - Cifar10 or Cifar100
  // @param const std::string &usage - Usage of this dataset, can be 'train', 'test' or 'all'
  // @param uint32_t numWorks - Num of workers reading images in parallel
  // @param std::string - dir directory of cifar dataset
  // @param uint32_t - queueSize - connector queue size
  // @param std::unique_ptr<Sampler> sampler - sampler tells ImageFolderOp what to read
  CifarOp(CifarType type, const std::string &usage, int32_t num_works, const std::string &file_dir, int32_t queue_size,
          std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler);
  // Destructor.
  ~CifarOp() = default;

  // A print method typically used for debugging
  // @param out
  // @param show_all
  void Print(std::ostream &out, bool show_all) const override;

  /// Function to count the number of samples in the CIFAR dataset
  /// @param dir path to the CIFAR directory
  /// @param isCIFAR10 true if CIFAR10 and false if CIFAR100
  /// @param count output arg that will hold the actual dataset size
  /// @return
  static Status CountTotalRows(const std::string &dir, const std::string &usage, bool isCIFAR10, int64_t *count);

  /// Op name getter
  /// @return Name of the current Op
  std::string Name() const override { return "CifarOp"; }

 private:
  // Load a tensor row according to a pair
  // @param uint64_t index - index need to load
  // @param TensorRow row - image & label read into this tensor row
  // @return Status The status code returned
  Status LoadTensorRow(row_id_type index, TensorRow *trow) override;

 private:
  // Read block data from cifar file
  // @return
  Status ReadCifarBlockDataAsync();

  // Called first when function is called
  // @return
  Status LaunchThreadsAndInitOp() override;

  /// Get cifar files in dir
  /// @return
  Status GetCifarFiles();

  /// Read cifar10 data as block
  /// @return
  Status ReadCifar10BlockData();

  /// Read cifar100 data as block
  /// @return
  Status ReadCifar100BlockData();

  /// Parse cifar data
  /// @return
  Status ParseCifarData();

  /// Method derived from RandomAccess Op, enable Sampler to get all ids for each class
  /// @param (std::map<uint32_t, std::vector<uint32_t >> *cls_ids - val all ids for this class
  /// @return Status The status code returned
  Status GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const override;

  /// Private function for computing the assignment of the column name map.
  /// @return - Status
  Status ComputeColMap() override;

  CifarType cifar_type_;
  std::string folder_path_;
  std::unique_ptr<DataSchema> data_schema_;

  const std::string usage_;  // can only be either "train" or "test"
  std::unique_ptr<Queue<std::vector<unsigned char>>> cifar_raw_data_block_;
  std::vector<std::string> cifar_files_;
  std::vector<std::string> path_record_;
  std::vector<std::pair<std::shared_ptr<Tensor>, std::vector<uint32_t>>> cifar_image_label_pairs_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  /// MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_CIFAR_OP_H_
