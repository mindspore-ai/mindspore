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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_LSUN_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_LSUN_OP_H_

#include <memory>
#include <queue>
#include <string>
#include <set>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/image_folder_op.h"

namespace mindspore {
namespace dataset {
/// \brief Forward declares.
template <typename T>
class Queue;

using ImageLabelPair = std::shared_ptr<std::pair<std::string, int32_t>>;
using FolderImagesPair = std::shared_ptr<std::pair<std::string, std::queue<ImageLabelPair>>>;

class LSUNOp : public ImageFolderOp {
 public:
  /// \brief Constructor.
  /// \param[in] int32_t num_wkrs num of workers reading images in parallel.
  /// \param[in] std::string file_dir dir directory of LSUNDataset.
  /// \param[in] int32_t queue_size connector queue size.
  /// \param[in] std::string usage Dataset splits of LSUN, can be `train`, `valid`, `test` or `all`.
  /// \param[in] std::vector<std::string> classes Classes list to load.
  /// \param[in] bool do_decode decode the images after reading.
  /// \param[in] std::unique_ptr<dataschema> data_schema schema of data.
  /// \param[in] unique_ptr<Sampler> sampler sampler tells LSUNOp what to read.
  LSUNOp(int32_t num_wkrs, const std::string &file_dir, int32_t queue_size, const std::string &usage,
         const std::vector<std::string> &classes, bool do_decode, std::unique_ptr<DataSchema> data_schema,
         std::shared_ptr<SamplerRT> sampler);

  /// \brief Destructor.
  ~LSUNOp() = default;

  /// \brief A print method typically used for debugging.
  /// \param[out] out The output stream to write output to.
  /// \param[in] show_all A bool to control if you want to show all info or just a summary.
  void Print(std::ostream &out, bool show_all) const override;

  /// \brief Function to count the number and classes of samples in the LSUN dataset.
  /// \param[in] const std::string &path path to the LSUN directory.
  /// \param[in] std::string usage Dataset splits of LSUN, can be `train`, `valid`, `test` or `all`.
  /// \param[in] const std::vector<std::string> &classes Classes list to load.
  /// \param[out] int64_t *num_rows output arg that will hold the minimum of the actual dataset
  ///     size and numSamples.
  /// \param[out] int64_t *num_classes output arg that will hold the classes num of the actual dataset
  ///     size and numSamples.
  /// \return Status The status code returned.
  static Status CountRowsAndClasses(const std::string &path, const std::string &usage,
                                    const std::vector<std::string> &classes, int64_t *num_rows, int64_t *num_classes);

  /// \brief Op name getter.
  /// \return Name of the current Op.
  std::string Name() const override { return "LSUNOp"; }

  /// \brief Dataset name getter.
  /// \param[in] upper Whether to get upper name.
  /// \return Dataset name of the current Op.
  std::string DatasetName(bool upper = false) const override { return upper ? "LSUN" : "lsun"; }

  /// \brief Load a tensor row according to a pair
  /// \param[in] row_id id for this tensor row
  /// \param[out] trow image & label read into this tensor row
  /// \return Status The status code returned
  Status LoadTensorRow(row_id_type row_id, TensorRow *trow) override;

  /// \brief Base-class override for GetNumClasses
  /// \param[out] num_classes the number of classes
  /// \return Status of the function
  Status GetNumClasses(int64_t *num_classes) override;

 private:
  /// \brief Base-class override for RecursiveWalkFolder
  /// \param[in] std::string & dir dir to lsun dataset.
  /// \return Status of the function
  Status RecursiveWalkFolder(Path *dir) override;

  /// \brief Function to save the path list to folder_paths
  /// \param[in] std::string & dir dir to lsun dataset.
  /// \param[in] std::string usage Dataset splits of LSUN, can be `train`, `valid`, `test` or `all`.
  /// \param[in] const std::vector<std::string> &classes Classes list to load.
  /// \param[out] std::unique_ptr<Queue<std::string>> &folder_name_queue output arg that will hold the path list.
  /// \param[out] int64_t *num_class the number of classes
  /// \return Status of the function
  static Status WalkDir(Path *dir, const std::string &usage, const std::vector<std::string> &classes,
                        const std::unique_ptr<Queue<std::string>> &folder_name_queue, int64_t *num_class);

  std::string usage_;
  std::vector<std::string> classes_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_LSUN_OP_H_
