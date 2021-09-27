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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_IMAGE_FOLDER_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_IMAGE_FOLDER_OP_H_

#include <deque>
#include <memory>
#include <queue>
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <utility>
#include <vector>
#include "minddata/dataset/core/tensor.h"

#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/engine/datasetops/source/mappable_leaf_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/image_utils.h"
#else
#include "minddata/dataset/kernels/image/lite_image_utils.h"
#endif
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/util/services.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/wait_post.h"

namespace mindspore {
namespace dataset {
/// Forward declares
template <typename T>
class Queue;

using ImageLabelPair = std::shared_ptr<std::pair<std::string, int32_t>>;
using FolderImagesPair = std::shared_ptr<std::pair<std::string, std::queue<ImageLabelPair>>>;

class ImageFolderOp : public MappableLeafOp {
 public:
  // Constructor
  // @param int32_t num_wkrs - Num of workers reading images in parallel
  // @param std::string - dir directory of ImageNetFolder
  // @param int32_t queue_size - connector queue size
  // @param bool recursive - read recursively
  // @param bool do_decode - decode the images after reading
  // @param std::set<std::string> &exts - set of file extensions to read, if empty, read everything under the dir
  // @param std::map<std::string, int32_t> &map- map of folder name and class id
  // @param std::unique_ptr<dataschema> data_schema - schema of data
  ImageFolderOp(int32_t num_wkrs, std::string file_dir, int32_t queue_size, bool recursive, bool do_decode,
                const std::set<std::string> &exts, const std::map<std::string, int32_t> &map,
                std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler);

  /// Destructor.
  ~ImageFolderOp() = default;

  /// Initialize ImageFOlderOp related var, calls the function to walk all files
  /// @param - std::string dir file directory to  ImageNetFolder
  /// @return Status The status code returned
  Status PrescanMasterEntry(const std::string &dir);

  // Worker thread pulls a number of IOBlock from IOBlock Queue, make a TensorRow and push it to Connector
  // @param int32_t workerId - id of each worker
  // @return Status The status code returned
  Status PrescanWorkerEntry(int32_t worker_id);

  // Method derived from RandomAccess Op, enable Sampler to get all ids for each class
  // @param (std::map<int64_t, std::vector<int64_t >> * map - key label, val all ids for this class
  // @return Status The status code returned
  Status GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const override;

  /// A print method typically used for debugging
  /// @param out
  /// @param show_all
  void Print(std::ostream &out, bool show_all) const override;

  /// This function is a hack! It is to return the num_class and num_rows. The result
  /// returned by this function may not be consistent with what image_folder_op is going to return
  /// user this at your own risk!
  static Status CountRowsAndClasses(const std::string &path, const std::set<std::string> &exts, int64_t *num_rows,
                                    int64_t *num_classes, std::map<std::string, int32_t> class_index);

  /// Op name getter
  /// @return Name of the current Op
  std::string Name() const override { return "ImageFolderOp"; }

  // DatasetName name getter
  // \return DatasetName of the current Op
  virtual std::string DatasetName(bool upper = false) const { return upper ? "ImageFolder" : "image folder"; }

  //// \brief Base-class override for GetNumClasses
  //// \param[out] num_classes the number of classes
  //// \return Status of the function
  Status GetNumClasses(int64_t *num_classes) override;

 private:
  // Load a tensor row according to a pair
  // @param row_id_type row_id - id for this tensor row
  // @param ImageLabelPair pair - <imagefile,label>
  // @param TensorRow row - image & label read into this tensor row
  // @return Status The status code returned
  Status LoadTensorRow(row_id_type row_id, TensorRow *row) override;

  /// @param std::string & dir - dir to walk all images
  /// @param int64_t * cnt - number of non folder files under the current dir
  /// @return
  Status RecursiveWalkFolder(Path *dir);

  /// start walking of all dirs
  /// @return
  Status StartAsyncWalk();

  // Called first when function is called
  // @return
  Status LaunchThreadsAndInitOp() override;

  /// Private function for computing the assignment of the column name map.
  /// @return - Status
  Status ComputeColMap() override;

  std::string folder_path_;  // directory of image folder
  bool recursive_;
  bool decode_;
  std::set<std::string> extensions_;  // extensions allowed
  std::map<std::string, int32_t> class_index_;
  std::unique_ptr<DataSchema> data_schema_;
  int64_t sampler_ind_;
  uint64_t dirname_offset_;
  std::vector<ImageLabelPair> image_label_pairs_;
  std::unique_ptr<Queue<std::string>> folder_name_queue_;
  std::unique_ptr<Queue<FolderImagesPair>> image_name_queue_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_IMAGE_FOLDER_OP_H_
