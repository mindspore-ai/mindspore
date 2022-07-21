/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/wait_post.h"

namespace mindspore {
namespace dataset {
// Forward declares
template <typename T>
class Queue;

using CoordinateRow = std::vector<std::vector<float>>;

class CocoOp : public MappableLeafOp {
 public:
  enum class TaskType { Detection = 0, Stuff = 1, Panoptic = 2, Keypoint = 3, Captioning = 4 };

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

#ifdef ENABLE_PYTHON
  /// \brief Constructor.
  /// \param[in] task_type Task type of Coco.
  /// \param[in] image_folder_path Image folder path of Coco.
  /// \param[in] annotation_path Annotation json path of Coco.
  /// \param[in] num_workers Number of workers reading images in parallel.
  /// \param[in] queue_size Connector queue size.
  /// \param[in] num_samples Number of samples to read.
  /// \param[in] decode Whether to decode images.
  /// \param[in] data_schema The schema of the Coco dataset.
  /// \param[in] sampler Sampler tells CocoOp what to read.
  /// \param[in] decrypt - Image decryption function, which accepts the path of the encrypted image file
  ///     and returns the decrypted bytes data. Default: None, no decryption.
  CocoOp(const TaskType &task_type, const std::string &image_folder_path, const std::string &annotation_path,
         int32_t num_workers, int32_t queue_size, bool decode, std::unique_ptr<DataSchema> data_schema,
         std::shared_ptr<SamplerRT> sampler, bool extra_metadata, py::function decrypt = py::none());
#else
  /// \brief Constructor.
  /// \param[in] task_type Task type of Coco.
  /// \param[in] image_folder_path Image folder path of Coco.
  /// \param[in] annotation_path Annotation json path of Coco.
  /// \param[in] num_workers Number of workers reading images in parallel.
  /// \param[in] queue_size Connector queue size.
  /// \param[in] num_samples Number of samples to read.
  /// \param[in] decode Whether to decode images.
  /// \param[in] data_schema The schema of the Coco dataset.
  /// \param[in] sampler Sampler tells CocoOp what to read.
  CocoOp(const TaskType &task_type, const std::string &image_folder_path, const std::string &annotation_path,
         int32_t num_workers, int32_t queue_size, bool decode, std::unique_ptr<DataSchema> data_schema,
         std::shared_ptr<SamplerRT> sampler, bool extra_metadata);
#endif

  /// \brief Destructor.
  ~CocoOp() = default;

  /// \brief A print method typically used for debugging.
  /// \param[out] out The output stream to write output to.
  /// \param[in] show_all A bool to control if you want to show all info or just a summary.
  void Print(std::ostream &out, bool show_all) const override;

  /// \param[out] count Output rows number of CocoDataset.
  Status CountTotalRows(int64_t *count);

  /// \brief Op name getter.
  /// \return Name of the current Op.
  std::string Name() const override { return "CocoOp"; }

  /// \brief Gets the class indexing.
  /// \return Status The status code returned.
  Status GetClassIndexing(std::vector<std::pair<std::string, std::vector<int32_t>>> *output_class_indexing) override;

 private:
  /// \brief Load a tensor row according to image id.
  /// \param[in] row_id Id for this tensor row.
  /// \param[out] row Image & target read into this tensor row.
  /// \return Status The status code returned.
  Status LoadTensorRow(row_id_type row_id, TensorRow *row) override;

  /// \brief Load a tensor row with vector which a vector to a tensor, for "Detection" task.
  /// \param[in] row_id Id for this tensor row.
  /// \param[in] image_id Image id.
  /// \param[in] image Image tensor.
  /// \param[in] coordinate Coordinate tensor.
  /// \param[out] row Image & target read into this tensor row.
  /// \return Status The status code returned.
  Status LoadDetectionTensorRow(row_id_type row_id, const std::string &image_id, std::shared_ptr<Tensor> image,
                                std::shared_ptr<Tensor> coordinate, TensorRow *trow);

  /// \brief Load a tensor row with vector which a vector to a tensor, for "Stuff/Keypoint" task.
  /// \param[in] row_id Id for this tensor row.
  /// \param[in] image_id Image id.
  /// \param[in] image Image tensor.
  /// \param[in] coordinate Coordinate tensor.
  /// \param[out] row Image & target read into this tensor row.
  /// \return Status The status code returned.
  Status LoadSimpleTensorRow(row_id_type row_id, const std::string &image_id, std::shared_ptr<Tensor> image,
                             std::shared_ptr<Tensor> coordinate, TensorRow *trow);

  /// \brief Load a tensor row with vector which a vector to multi-tensor, for "Panoptic" task.
  /// \param[in] row_id Id for this tensor row.
  /// \param[in] image_id Image id.
  /// \param[in] image Image tensor.
  /// \param[in] coordinate Coordinate tensor.
  /// \param[out] row Image & target read into this tensor row.
  /// \return Status The status code returned.
  Status LoadMixTensorRow(row_id_type row_id, const std::string &image_id, std::shared_ptr<Tensor> image,
                          std::shared_ptr<Tensor> coordinate, TensorRow *trow);

  /// \brief Load a tensor row with vector which a vector to multi-tensor, for "Captioning" task.
  /// \param[in] row_id Id for this tensor row.
  /// \param[in] image_id Image id.
  /// \param[in] image Image tensor.
  /// \param[in] captions Captions tensor.
  /// \param[out] trow Image & target read into this tensor row.
  /// \return Status The status code returned.
  Status LoadCaptioningTensorRow(row_id_type row_id, const std::string &image_id, std::shared_ptr<Tensor> image,
                                 std::shared_ptr<Tensor> captions, TensorRow *trow);

  /// \param[in] path Path to the image file.
  /// \param[out] tensor Returned tensor.
  /// \return Status The status code returned.
  Status ReadImageToTensor(const std::string &path, std::shared_ptr<Tensor> *tensor) const;

  /// \brief Read annotation from Annotation folder.
  /// \return Status The status code returned.
  Status PrepareData() override;

  /// \param[in] image_tree Image tree of json.
  /// \param[out] image_vec Image id list of json.
  /// \return Status The status code returned.
  Status ImageColumnLoad(const nlohmann::json &image_tree, std::vector<std::string> *image_vec);

  /// \param[in] categories_tree Categories tree of json.
  /// \return Status The status code returned.
  Status CategoriesColumnLoad(const nlohmann::json &categories_tree);

  /// \param[in] categories_tree Categories tree of json.
  /// \param[in] image_file Current image name in annotation.
  /// \param[in] id Current unique id of annotation.
  /// \return Status The status code returned.
  Status DetectionColumnLoad(const nlohmann::json &annotation_tree, const std::string &image_file, const int32_t &id);

  /// \param[in] categories_tree Categories tree of json.
  /// \param[in] image_file Current image name in annotation.
  /// \param[in] id Current unique id of annotation.
  /// \return Status The status code returned.
  Status StuffColumnLoad(const nlohmann::json &annotation_tree, const std::string &image_file, const int32_t &id);

  /// \param[in] categories_tree Categories tree of json.
  /// \param[in] image_file Current image name in annotation.
  /// \param[in] id Current unique id of annotation.
  /// \return Status The status code returned.
  Status KeypointColumnLoad(const nlohmann::json &annotation_tree, const std::string &image_file, const int32_t &id);

  /// \param[in] categories_tree Categories tree of json.
  /// \param[in] image_file Current image name in annotation.
  /// \param[in] image_id Current unique id of annotation.
  /// \return Status The status code returned.
  Status PanopticColumnLoad(const nlohmann::json &annotation_tree, const std::string &image_file,
                            const int32_t &image_id);

  /// \brief Function for finding a caption in annotation_tree.
  /// \param[in] annotation_tree Annotation tree of json.
  /// \param[in] image_file Current image name in annotation.
  /// \param[in] id Current unique id of annotation.
  /// \return Status The status code returned.
  Status CaptionColumnLoad(const nlohmann::json &annotation_tree, const std::string &image_file, const int32_t &id);

  template <typename T>
  Status SearchNodeInJson(const nlohmann::json &input_tree, std::string node_name, T *output_node);

  /// \brief Private function for computing the assignment of the column name map.
  /// \return Status The status code returned.
  Status ComputeColMap() override;

  bool decode_;
  std::string image_folder_path_;
  std::string annotation_path_;
  TaskType task_type_;
  std::unique_ptr<DataSchema> data_schema_;
  bool extra_metadata_;

  std::vector<std::string> image_ids_;
  std::map<int32_t, std::string> image_index_;
  std::vector<std::pair<std::string, std::vector<int32_t>>> label_index_;
  std::map<std::string, CoordinateRow> coordinate_map_;
  std::map<std::string, std::vector<uint32_t>> simple_item_map_;
  std::map<std::string, std::vector<std::string>> captions_map_;
  std::set<uint32_t> category_set_;
#ifdef ENABLE_PYTHON
  py::function decrypt_;
#endif
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_COCO_OP_H_
