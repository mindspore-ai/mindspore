/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_RENDERED_SST2_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_RENDERED_SST2_OP_H_

#include <algorithm>
#include <deque>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <string>
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

using ImageLabelPair = std::shared_ptr<std::pair<std::string, uint32_t>>;
using FolderImagesPair = std::shared_ptr<std::pair<std::string, std::queue<ImageLabelPair>>>;

class RenderedSST2Op : public MappableLeafOp {
 public:
  /// Constructor.
  /// @param int32_t num_wkrs - Num of workers reading images in parallel.
  /// @param const std::string &file_dir - Directory of RenderedSST2Dataset.
  /// @param const std::string &usage -  Usage of this dataset, can be 'train', 'test', 'val' or 'all'.
  /// @param int32_t queue_size - Connector queue size.
  /// @param bool do_decode - Decode the images after reading.
  /// @param std::set<std::string> &exts - Set of file extensions to read, if empty, read everything under the dir.
  /// @param std::map<std::string, int32_t> &map- Map of class name and class id.
  /// @param std::unique_ptr<dataschema> data_schema - Schema of data.
  /// @param std::shared_ptr<SamplerRT> sampler - Sampler tells RenderedSST2Op what to read.
  RenderedSST2Op(int32_t num_wkrs, const std::string &file_dir, const std::string &usage, int32_t queue_size,
                 bool do_decode, const std::set<std::string> &exts, const std::map<std::string, uint32_t> &map,
                 std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler);

  /// Destructor.
  ~RenderedSST2Op() override = default;

  /// Initialize RenderedSST2Op related var, calls the function to walk all files.
  /// @return Status The status code returned.
  Status PrepareData() override;

  /// Worker thread pulls a number of IOBlock from IOBlock Queue, make a TensorRow and push it to Connector.
  /// @param int32_t worker_id - Id of each worker.
  /// @return Status The status code returned.
  Status PrescanWorkerEntry(int32_t worker_id);

  /// Method derived from RandomAccess Op, enable Sampler to get all ids for each class.
  /// @param (std::map<int32_t, std::vector<int64_t >> * cls_ids - Key label, val all ids for this class.
  /// @return Status The status code returned.
  Status GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const override;

  /// A print method typically used for debugging.
  /// @param std::ostream &out - Out stream.
  /// @param bool show_all - Whether to show all information.
  void Print(std::ostream &out, bool show_all) const override;

  /// This function is a hack! It is to return the num_class and num_rows. The result
  /// returned by this function may not be consistent with what RenderedSST2Op is going to return
  /// user this at your own risk!
  /// @param const std::string &path - Directory of RenderedSST2Dataset.
  /// @param const std::string &usage - Usage of this dataset, can be 'train', 'test', 'valid' or 'all'.
  /// @param const std::set<std::string> &exts - Set of file extensions to read, if empty, read everything under the
  ///     dir.
  /// @param int64_t *num_rows - The number of rows.
  /// @param int64_t *num_classes - The number of classes.
  /// @return Status of the function.
  static Status CountRowsAndClasses(const std::string &path, const std::string &usage,
                                    const std::set<std::string> &exts, int64_t *num_rows, int64_t *num_classes);

  /// This help function is used to count the num_rows.
  /// @param std::queue<std::string> *folder_paths - A queue contains all the image folder paths.
  /// @param int64_t *num_rows - The number of rows.
  /// @param const std::set<std::string> &exts - Set of file extensions to read, if empty, read everything under the
  ///     dir.
  /// @return Status of the function.
  static Status CountRows(std::queue<std::string> *folder_paths, int64_t *num_rows, const std::set<std::string> &exts);

  /// Op name getter.
  /// @return Name of the current Op.
  std::string Name() const override { return "RenderedSST2Op"; }

  /// DatasetName name getter.
  /// @return DatasetName of the current Op.
  virtual std::string DatasetName(bool upper = false) const { return upper ? "RenderedSST2" : "rendered sst2"; }

  /// Base-class override for GetNumClasses.
  /// @param num_classes - The number of classes.
  /// @return Status of the function.
  Status GetNumClasses(int64_t *num_classes) override;

 protected:
  /// Load a tensor row according to a pair.
  /// @param row_id_type row_id - Id for this tensor row.
  /// @param TensorRow *row - Image & label read into this tensor row.
  /// @return Status The status code returned.
  Status LoadTensorRow(row_id_type row_id, TensorRow *row) override;

  /// @param Path * dir - Dir to walk all folders.
  /// @return Status The status code returned.
  virtual Status WalkFolder(Path *dir);

  /// @param Path * dir - Dir to walk all images.
  /// @param std::queue<std::string> *folder_paths - A queue contains all the image folder paths.
  /// @param std::map<std::string, int32_t> *class_index - A map records the class and the class's Id.
  /// @return Status The status code returned.
  static Status WalkFolderForCountRows(Path *dir, std::queue<std::string> *folder_paths,
                                       std::map<std::string, uint32_t> *class_index);

  /// start walking of all dirs.
  /// @return Status The status code returned.
  Status StartAsyncWalk();

  /// Called first when function is called.
  /// @return Status The status code returned.
  Status RegisterAndLaunchThreads() override;

  /// Private function for computing the assignment of the column name map.
  /// @return Status The status code returned.
  Status ComputeColMap() override;

  std::string folder_path_;  // directory of image folder
  std::string usage_;
  bool decode_;
  std::set<std::string> extensions_;  // extensions allowed
  std::map<std::string, uint32_t> class_index_;
  std::unique_ptr<DataSchema> data_schema_;
  int64_t sampler_ind_;
  std::vector<ImageLabelPair> image_label_pairs_;
  std::vector<std::string> image_prefix_;
  std::unique_ptr<Queue<std::string>> folder_path_queue_;
  std::unique_ptr<Queue<uint32_t>> folder_classId_queue_;  // the class Id of the images under the folder
  std::unique_ptr<Queue<FolderImagesPair>> image_name_queue_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_RENDERED_SST2_OP_H_
