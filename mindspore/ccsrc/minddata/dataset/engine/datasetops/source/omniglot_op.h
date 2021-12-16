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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_OMNIGLOT_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_OMNIGLOT_OP_H_

#include <memory>
#include <queue>
#include <string>
#include <utility>

#include "minddata/dataset/engine/datasetops/source/image_folder_op.h"

namespace mindspore {
namespace dataset {
// Forward declares.
template <typename T>
class Queue;

using ImageLabelPair = std::shared_ptr<std::pair<std::string, int32_t>>;
using FolderImagesPair = std::shared_ptr<std::pair<std::string, std::queue<ImageLabelPair>>>;

class OmniglotOp : public ImageFolderOp {
 public:
  /// Constructor
  /// @param num_wkrs - Num of workers reading images in parallel.
  /// @param file_dir - Directory of ImageNetFolder.
  /// @param queue_size - Connector queue size.
  /// @param background - Use the background dataset or the evaluation dataset.
  /// @param do_decode - Decode the images after reading.
  /// @param data_schema - Schema of Omniglot dataset.
  /// @param sampler - Sampler tells OmniglotOp what to read.
  OmniglotOp(int32_t num_wkrs, const std::string &file_dir, int32_t queue_size, bool background, bool do_decode,
             std::unique_ptr<DataSchema> data_schema, const std::shared_ptr<SamplerRT> &sampler);

  /// Destructor.
  ~OmniglotOp() = default;

  /// A print method typically used for debugging.
  /// @param out - The output stream to write output to.
  /// @param show_all - A bool to control if you want to show all info or just a summary.
  void Print(std::ostream &out, bool show_all) const override;

  /// This is the common function to walk one directory.
  /// @param dir - The directory path
  /// @param folder_path - The queue in CountRowsAndClasses function.
  /// @param folder_name_queue - The queue in base class.
  /// @param dirname_offset - The offset of path of directory using in RecursiveWalkFolder function.
  /// @param std_queue - A bool to use folder_path or the foler_name_queue.
  /// @return Status - The error code returned.
  static Status WalkDir(Path *dir, std::queue<std::string> *folder_paths, Queue<std::string> *folder_name_queue,
                        uint64_t dirname_offset, bool std_queue);

  /// This function is a hack! It is to return the num_class and num_rows. The result
  /// returned by this function may not be consistent with what omniglot_op is going to return
  /// use this at your own risk!
  /// @param path - The folder path
  /// @param num_rows - The point to the number of rows
  /// @param num_classes - The point to the number of classes
  /// @return Status - the error code returned.
  static Status CountRowsAndClasses(const std::string &path, int64_t *num_rows, int64_t *num_classes);

  /// Op name getter
  /// @return std::string - Name of the current Op.
  std::string Name() const override { return "OmniglotOp"; }

  /// DatasetName name getter
  /// @param upper - A bool to control if you want to return uppercase or lowercase Op name.
  /// @return std::string - DatasetName of the current Op
  std::string DatasetName(bool upper = false) const { return upper ? "Omniglot" : "omniglot"; }

  /// Base-class override for GetNumClasses.
  /// @param num_classes - the number of classes.
  /// @return Status - the error code returned.
  Status GetNumClasses(int64_t *num_classes) override;

 private:
  //  Walk the folder
  /// @param dir - The folder path
  /// @return Status - the error code returned.
  Status RecursiveWalkFolder(Path *dir) override;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_OMNIGLOT_OP_H_
