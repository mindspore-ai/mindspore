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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_COCO_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_COCO_OP_H_

#include <map>
#include <memory>
#include <set>
#include <string>
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
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/wait_post.h"

namespace mindspore {
namespace dataset {
// Forward declares
template <typename T>
class Queue;

using CoordinateRow = std::vector<std::vector<float>>;

class CocoOp : public ParallelOp, public RandomAccessOp {
 public:
  enum class TaskType { Detection = 0, Stuff = 1, Panoptic = 2, Keypoint = 3 };

  class Builder {
   public:
    // Constructor for Builder class of ImageFolderOp
    // @param  uint32_t numWrks - number of parallel workers
    // @param dir - directory folder got ImageNetFolder
    Builder();

    // Destructor.
    ~Builder() = default;

    // Setter method.
    // @param const std::string & build_dir
    // @return Builder setter method returns reference to the builder.
    Builder &SetDir(const std::string &build_dir) {
      builder_dir_ = build_dir;
      return *this;
    }

    // Setter method.
    // @param const std::string & build_file
    // @return Builder setter method returns reference to the builder.
    Builder &SetFile(const std::string &build_file) {
      builder_file_ = build_file;
      return *this;
    }

    // Setter method.
    // @param const std::string & task_type
    // @return Builder setter method returns reference to the builder.
    Builder &SetTask(const std::string &task_type) {
      if (task_type == "Detection") {
        builder_task_type_ = TaskType::Detection;
      } else if (task_type == "Stuff") {
        builder_task_type_ = TaskType::Stuff;
      } else if (task_type == "Panoptic") {
        builder_task_type_ = TaskType::Panoptic;
      } else if (task_type == "Keypoint") {
        builder_task_type_ = TaskType::Keypoint;
      }
      return *this;
    }

    // Setter method.
    // @param int32_t num_workers
    // @return Builder setter method returns reference to the builder.
    Builder &SetNumWorkers(int32_t num_workers) {
      builder_num_workers_ = num_workers;
      return *this;
    }

    // Setter method.
    // @param int32_t op_connector_size
    // @return Builder setter method returns reference to the builder.
    Builder &SetOpConnectorSize(int32_t op_connector_size) {
      builder_op_connector_size_ = op_connector_size;
      return *this;
    }

    // Setter method.
    // @param int32_t rows_per_buffer
    // @return Builder setter method returns reference to the builder.
    Builder &SetRowsPerBuffer(int32_t rows_per_buffer) {
      builder_rows_per_buffer_ = rows_per_buffer;
      return *this;
    }

    // Setter method.
    // @param std::shared_ptr<Sampler> sampler
    // @return Builder setter method returns reference to the builder.
    Builder &SetSampler(std::shared_ptr<SamplerRT> sampler) {
      builder_sampler_ = std::move(sampler);
      return *this;
    }

    // Setter method.
    // @param bool do_decode
    // @return Builder setter method returns reference to the builder.
    Builder &SetDecode(bool do_decode) {
      builder_decode_ = do_decode;
      return *this;
    }

    // Check validity of input args
    // @return Status The status code returned
    Status SanityCheck();

    // The builder "Build" method creates the final object.
    // @param std::shared_ptr<CocoOp> *op - DatasetOp
    // @return Status The status code returned
    Status Build(std::shared_ptr<CocoOp> *op);

   private:
    bool builder_decode_;
    std::string builder_dir_;
    std::string builder_file_;
    TaskType builder_task_type_;
    int32_t builder_num_workers_;
    int32_t builder_op_connector_size_;
    int32_t builder_rows_per_buffer_;
    std::shared_ptr<SamplerRT> builder_sampler_;
    std::unique_ptr<DataSchema> builder_schema_;
  };

  // Constructor
  // @param TaskType task_type - task type of Coco
  // @param std::string image_folder_path - image folder path of Coco
  // @param std::string annotation_path - annotation json path of Coco
  // @param int32_t num_workers - number of workers reading images in parallel
  // @param int32_t rows_per_buffer - number of images (rows) in each buffer
  // @param int32_t queue_size - connector queue size
  // @param int64_t num_samples - number of samples to read
  // @param bool decode - whether to decode images
  // @param std::unique_ptr<DataSchema> data_schema - the schema of the Coco dataset
  // @param std::shared_ptr<Sampler> sampler - sampler tells CocoOp what to read
  CocoOp(const TaskType &task_type, const std::string &image_folder_path, const std::string &annotation_path,
         int32_t num_workers, int32_t rows_per_buffer, int32_t queue_size, bool decode,
         std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler);

  // Destructor
  ~CocoOp() = default;

  // Worker thread pulls a number of IOBlock from IOBlock Queue, make a buffer and push it to Connector
  // @param int32_t workerId - id of each worker
  // @return Status The status code returned
  Status WorkerEntry(int32_t worker_id) override;

  // Main Loop of CocoOp
  // Master thread: Fill IOBlockQueue, then goes to sleep
  // Worker thread: pulls IOBlock from IOBlockQueue, work on it the put buffer to mOutConnector
  // @return Status The status code returned
  Status operator()() override;

  // A print method typically used for debugging
  // @param out
  // @param show_all
  void Print(std::ostream &out, bool show_all) const override;

  // @param const std::string &dir - Coco image dir path
  // @param const std::string &file - Coco json file path
  // @param const std::string &task - task mode of Coco task
  // @param int64_t numSamples - samples number of CocoDataset
  // @param int64_t *count - output rows number of CocoDataset
  static Status CountTotalRows(const std::string &dir, const std::string &task_type, const std::string &task_mode,
                               int64_t *count);

  // @param const std::string &dir - Coco image dir path
  // @param const std::string &file - Coco json file path
  // @param const std::string &task - task mode of Coco task
  // @param int64_t numSamples - samples number of CocoDataset
  // @param std::map<std::string, int32_t> *output_class_indexing - output class index of CocoDataset
  static Status GetClassIndexing(const std::string &dir, const std::string &task_type, const std::string &task_mode,
                                 std::vector<std::pair<std::string, std::vector<int32_t>>> *output_class_indexing);

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return "CocoOp"; }

  /// \brief Gets the class indexing
  /// \return Status The status code returned
  Status GetClassIndexing(std::vector<std::pair<std::string, std::vector<int32_t>>> *output_class_indexing) override;

 private:
  // Initialize Sampler, calls sampler->Init() within
  // @return Status The status code returned
  Status InitSampler();

  // Load a tensor row according to image id
  // @param row_id_type row_id - id for this tensor row
  // @param std::string image_id - image id
  // @param TensorRow row - image & target read into this tensor row
  // @return Status The status code returned
  Status LoadTensorRow(row_id_type row_id, const std::string &image_id, TensorRow *row);

  // Load a tensor row with vector which a vector to a tensor
  // @param row_id_type row_id - id for this tensor row
  // @param const std::string &image_id - image is
  // @param std::shared_ptr<Tensor> image - image tensor
  // @param std::shared_ptr<Tensor> coordinate - coordinate tensor
  // @param TensorRow row - image & target read into this tensor row
  // @return Status The status code returned
  Status LoadDetectionTensorRow(row_id_type row_id, const std::string &image_id, std::shared_ptr<Tensor> image,
                                std::shared_ptr<Tensor> coordinate, TensorRow *trow);

  // Load a tensor row with vector which a vector to a tensor
  // @param row_id_type row_id - id for this tensor row
  // @param const std::string &image_id - image is
  // @param std::shared_ptr<Tensor> image - image tensor
  // @param std::shared_ptr<Tensor> coordinate - coordinate tensor
  // @param TensorRow row - image & target read into this tensor row
  // @return Status The status code returned
  Status LoadSimpleTensorRow(row_id_type row_id, const std::string &image_id, std::shared_ptr<Tensor> image,
                             std::shared_ptr<Tensor> coordinate, TensorRow *trow);

  // Load a tensor row with vector which a vector to multi-tensor
  // @param row_id_type row_id - id for this tensor row
  // @param const std::string &image_id - image is
  // @param std::shared_ptr<Tensor> image - image tensor
  // @param std::shared_ptr<Tensor> coordinate - coordinate tensor
  // @param TensorRow row - image & target read into this tensor row
  // @return Status The status code returned
  Status LoadMixTensorRow(row_id_type row_id, const std::string &image_id, std::shared_ptr<Tensor> image,
                          std::shared_ptr<Tensor> coordinate, TensorRow *trow);

  // @param const std::string &path - path to the image file
  // @param const ColDescriptor &col - contains tensor implementation and datatype
  // @param std::shared_ptr<Tensor> tensor - return
  // @return Status The status code returned
  Status ReadImageToTensor(const std::string &path, const ColDescriptor &col, std::shared_ptr<Tensor> *tensor);

  // @param const std::vector<uint64_t> &keys - keys in ioblock
  // @param std::unique_ptr<DataBuffer> db
  // @return Status The status code returned
  Status LoadBuffer(const std::vector<int64_t> &keys, std::unique_ptr<DataBuffer> *db);

  // Read annotation from Annotation folder
  // @return Status The status code returned
  Status ParseAnnotationIds();

  // @param const std::shared_ptr<Tensor> &sample_ids - sample ids of tensor
  // @param std::vector<int64_t> *keys - image id
  // @return Status The status code returned
  Status TraverseSampleIds(const std::shared_ptr<Tensor> &sample_ids, std::vector<int64_t> *keys);

  // Called first when function is called
  // @return Status The status code returned
  Status LaunchThreadsAndInitOp();

  // Reset dataset state
  // @return Status The status code returned
  Status Reset() override;

  // @param nlohmann::json image_tree - image tree of json
  // @param std::vector<std::string> *image_vec - image id list of json
  // @return Status The status code returned
  Status ImageColumnLoad(const nlohmann::json &image_tree, std::vector<std::string> *image_vec);

  // @param nlohmann::json categories_tree - categories tree of json
  // @return Status The status code returned
  Status CategoriesColumnLoad(const nlohmann::json &categories_tree);

  // @param nlohmann::json categories_tree - categories tree of json
  // @param const std::string &image_file - current image name in annotation
  // @param const int32_t &id - current unique id of annotation
  // @return Status The status code returned
  Status DetectionColumnLoad(const nlohmann::json &annotation_tree, const std::string &image_file, const int32_t &id);

  // @param nlohmann::json categories_tree - categories tree of json
  // @param const std::string &image_file - current image name in annotation
  // @param const int32_t &id - current unique id of annotation
  // @return Status The status code returned
  Status StuffColumnLoad(const nlohmann::json &annotation_tree, const std::string &image_file, const int32_t &id);

  // @param nlohmann::json categories_tree - categories tree of json
  // @param const std::string &image_file - current image name in annotation
  // @param const int32_t &id - current unique id of annotation
  // @return Status The status code returned
  Status KeypointColumnLoad(const nlohmann::json &annotation_tree, const std::string &image_file, const int32_t &id);

  // @param nlohmann::json categories_tree - categories tree of json
  // @param const std::string &image_file - current image name in annotation
  // @param const int32_t &image_id - current unique id of annotation
  // @return Status The status code returned
  Status PanopticColumnLoad(const nlohmann::json &annotation_tree, const std::string &image_file,
                            const int32_t &image_id);

  template <typename T>
  Status SearchNodeInJson(const nlohmann::json &input_tree, std::string node_name, T *output_node);

  // Private function for computing the assignment of the column name map.
  // @return - Status
  Status ComputeColMap() override;

  bool decode_;
  int64_t row_cnt_;
  int64_t buf_cnt_;
  std::string image_folder_path_;
  std::string annotation_path_;
  TaskType task_type_;
  int32_t rows_per_buffer_;
  std::unique_ptr<DataSchema> data_schema_;

  std::vector<std::string> image_ids_;
  std::map<int32_t, std::string> image_index_;
  std::vector<std::pair<std::string, std::vector<int32_t>>> label_index_;
  std::map<std::string, CoordinateRow> coordinate_map_;
  std::map<std::string, std::vector<uint32_t>> simple_item_map_;
  std::set<uint32_t> category_set_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_COCO_OP_H_
