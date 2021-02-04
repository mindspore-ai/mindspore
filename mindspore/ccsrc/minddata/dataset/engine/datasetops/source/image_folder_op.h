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
#include "minddata/dataset/engine/data_buffer.h"
#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
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
// Forward declares
template <typename T>
class Queue;

using ImageLabelPair = std::shared_ptr<std::pair<std::string, int32_t>>;
using FolderImagesPair = std::shared_ptr<std::pair<std::string, std::queue<ImageLabelPair>>>;

class ImageFolderOp : public ParallelOp, public RandomAccessOp {
 public:
  class Builder {
   public:
    // Constructor for Builder class of ImageFolderOp
    // @param  int32_t numWrks - number of parallel workers
    // @param dir - directory folder got ImageNetFolder
    Builder();

    // Destructor.
    ~Builder() = default;

    // Setter method
    // @param int32_t rows_per_buffer
    // @return Builder setter method returns reference to the builder.
    Builder &SetRowsPerBuffer(int32_t rows_per_buffer) {
      builder_rows_per_buffer_ = rows_per_buffer;
      return *this;
    }

    // Setter method
    // @param int32_t size
    // @return Builder setter method returns reference to the builder.
    Builder &SetOpConnectorSize(int32_t size) {
      builder_op_connector_size_ = size;
      return *this;
    }

    // Setter method
    // @param std::set<std::string> & exts, file extensions to be read
    // @return Builder setter method returns reference to the builder.
    Builder &SetExtensions(const std::set<std::string> &exts) {
      builder_extensions_ = exts;
      return *this;
    }

    // Setter method
    // @paramconst std::map<std::string, int32_t>& map - a class name to label map
    // @return
    Builder &SetClassIndex(const std::map<std::string, int32_t> &map) {
      builder_labels_to_read_ = map;
      return *this;
    }

    // Setter method
    // @param bool do_decode
    // @return Builder setter method returns reference to the builder.
    Builder &SetDecode(bool do_decode) {
      builder_decode_ = do_decode;
      return *this;
    }

    // Setter method
    // @param int32_t num_workers
    // @return Builder setter method returns reference to the builder.
    Builder &SetNumWorkers(int32_t num_workers) {
      builder_num_workers_ = num_workers;
      return *this;
    }

    // Setter method
    // @param std::shared_ptr<Sampler> sampler
    // @return Builder setter method returns reference to the builder.
    Builder &SetSampler(std::shared_ptr<SamplerRT> sampler) {
      builder_sampler_ = std::move(sampler);
      return *this;
    }

    // Setter method
    // @param const std::string & dir
    // @return
    Builder &SetImageFolderDir(const std::string &dir) {
      builder_dir_ = dir;
      return *this;
    }

    // Whether dir are walked recursively
    // @param bool recursive - if set to false, only get dirs in top level dir
    // @return
    Builder &SetRecursive(bool recursive) {
      builder_recursive_ = recursive;
      return *this;
    }

    // Check validity of input args
    // @return Status The status code returned
    Status SanityCheck();

    // The builder "build" method creates the final object.
    // @param std::shared_ptr<ImageFolderOp> *op - DatasetOp
    // @return Status The status code returned
    Status Build(std::shared_ptr<ImageFolderOp> *op);

   private:
    bool builder_decode_;
    bool builder_recursive_;
    std::string builder_dir_;
    int32_t builder_num_workers_;
    int32_t builder_rows_per_buffer_;
    int32_t builder_op_connector_size_;
    std::set<std::string> builder_extensions_;
    std::shared_ptr<SamplerRT> builder_sampler_;
    std::unique_ptr<DataSchema> builder_schema_;
    std::map<std::string, int32_t> builder_labels_to_read_;
  };

  // Constructor
  // @param int32_t num_wkrs - Num of workers reading images in parallel
  // @param int32_t - rows_per_buffer Number of images (rows) in each buffer
  // @param std::string - dir directory of ImageNetFolder
  // @param int32_t queue_size - connector queue size
  // @param std::set<std::string> exts - set of file extensions to read, if empty, read everything under the dir
  // @param td::unique_ptr<Sampler> sampler - sampler tells ImageFolderOp what to read
  ImageFolderOp(int32_t num_wkrs, int32_t rows_per_buffer, std::string file_dir, int32_t queue_size, bool recursive,
                bool do_decode, const std::set<std::string> &exts, const std::map<std::string, int32_t> &map,
                std::unique_ptr<DataSchema>, std::shared_ptr<SamplerRT> sampler);

  // Destructor.
  ~ImageFolderOp() = default;

  // Initialize ImageFOlderOp related var, calls the function to walk all files
  // @param - std::string dir file directory to  ImageNetFolder
  // @return Status The status code returned
  Status PrescanMasterEntry(const std::string &dir);

  // Worker thread pulls a number of IOBlock from IOBlock Queue, make a buffer and push it to Connector
  // @param int32_t workerId - id of each worker
  // @return Status The status code returned
  Status WorkerEntry(int32_t worker_id) override;

  // Worker thread pulls a number of IOBlock from IOBlock Queue, make a buffer and push it to Connector
  // @param int32_t workerId - id of each worker
  // @return Status The status code returned
  Status PrescanWorkerEntry(int32_t worker_id);

  // Main Loop of ImageFolderOp
  // Master thread: Fill IOBlockQueue, then goes to sleep
  // Worker thread: pulls IOBlock from IOBlockQueue, work on it then put buffer to mOutConnector
  // @return Status The status code returned
  Status operator()() override;

  // Method derived from RandomAccess Op, enable Sampler to get all ids for each class
  // @param (std::map<int64_t, std::vector<int64_t >> * map - key label, val all ids for this class
  // @return Status The status code returned
  Status GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const override;

  // A print method typically used for debugging
  // @param out
  // @param show_all
  void Print(std::ostream &out, bool show_all) const override;

  // This function is a hack! It is to return the num_class and num_rows. The result
  // returned by this function may not be consistent with what image_folder_op is going to return
  // user this at your own risk!
  static Status CountRowsAndClasses(const std::string &path, const std::set<std::string> &exts, int64_t *num_rows,
                                    int64_t *num_classes, std::map<std::string, int32_t> class_index);

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return "ImageFolderOp"; }

  /// \brief Base-class override for GetNumClasses
  /// \param[out] num_classes the number of classes
  /// \return Status of the function
  Status GetNumClasses(int64_t *num_classes) override;

 private:
  // Initialize Sampler, calls sampler->Init() within
  // @return Status The status code returned
  Status InitSampler();

  // Load a tensor row according to a pair
  // @param row_id_type row_id - id for this tensor row
  // @param ImageLabelPair pair - <imagefile,label>
  // @param TensorRow row - image & label read into this tensor row
  // @return Status The status code returned
  Status LoadTensorRow(row_id_type row_id, ImageLabelPair pair, TensorRow *row);

  // @param const std::vector<int64_t> &keys - keys in ioblock
  // @param std::unique_ptr<DataBuffer> db
  // @return Status The status code returned
  Status LoadBuffer(const std::vector<int64_t> &keys, std::unique_ptr<DataBuffer> *db);

  // @param std::string & dir - dir to walk all images
  // @param int64_t * cnt - number of non folder files under the current dir
  // @return
  Status RecursiveWalkFolder(Path *dir);

  // start walking of all dirs
  // @return
  Status StartAsyncWalk();

  // Called first when function is called
  // @return
  Status LaunchThreadsAndInitOp();

  // reset Op
  // @return Status The status code returned
  Status Reset() override;

  // Private function for computing the assignment of the column name map.
  // @return - Status
  Status ComputeColMap() override;

  int32_t rows_per_buffer_;
  std::string folder_path_;  // directory of image folder
  bool recursive_;
  bool decode_;
  std::set<std::string> extensions_;  // extensions allowed
  std::map<std::string, int32_t> class_index_;
  std::unique_ptr<DataSchema> data_schema_;
  int64_t row_cnt_;
  int64_t buf_cnt_;
  int64_t sampler_ind_;
  int64_t dirname_offset_;
  std::vector<ImageLabelPair> image_label_pairs_;
  std::unique_ptr<Queue<std::string>> folder_name_queue_;
  std::unique_ptr<Queue<FolderImagesPair>> image_name_queue_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_IMAGE_FOLDER_OP_H_
