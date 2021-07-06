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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_VOC_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_VOC_OP_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "./tinyxml2.h"
#include "minddata/dataset/core/tensor.h"

#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/engine/datasetops/source/mappable_leaf_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/wait_post.h"

using tinyxml2::XMLDocument;
using tinyxml2::XMLElement;
using tinyxml2::XMLError;
namespace mindspore {
namespace dataset {
// Forward declares
template <typename T>
class Queue;

using Annotation = std::vector<std::pair<std::string, std::vector<float>>>;

class VOCOp : public MappableLeafOp {
 public:
  enum class TaskType { Segmentation = 0, Detection = 1 };

  // Constructor
  // @param TaskType task_type - task type of VOC
  // @param std::string task_mode - task mode of VOC
  // @param std::string folder_path - dir directory of VOC
  // @param std::map<std::string, int32_t> class_index - input class-to-index of annotation
  // @param int32_t num_workers - number of workers reading images in parallel
  // @param int32_t queue_size - connector queue size
  // @param bool decode - whether to decode images
  // @param std::unique_ptr<DataSchema> data_schema - the schema of the VOC dataset
  // @param std::shared_ptr<Sampler> sampler - sampler tells VOCOp what to read
  // @param extra_metadata - flag to add extra meta-data to row
  VOCOp(const TaskType &task_type, const std::string &task_mode, const std::string &folder_path,
        const std::map<std::string, int32_t> &class_index, int32_t num_workers, int32_t queue_size, bool decode,
        std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler, bool extra_metadata);

  // Destructor
  ~VOCOp() = default;

  // A print method typically used for debugging
  // @param out
  // @param show_all
  void Print(std::ostream &out, bool show_all) const override;

  // @param int64_t *count - output rows number of VOCDataset
  Status CountTotalRows(int64_t *count);

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return "VOCOp"; }

  // /// \brief Gets the class indexing
  // /// \return Status - The status code return
  Status GetClassIndexing(std::vector<std::pair<std::string, std::vector<int32_t>>> *output_class_indexing) override;

 private:
  // Load a tensor row according to image id
  // @param row_id_type row_id - id for this tensor row
  // @param std::string image_id - image id
  // @param TensorRow row - image & target read into this tensor row
  // @return Status The status code returned
  Status LoadTensorRow(row_id_type row_id, TensorRow *row) override;

  // @param const std::string &path - path to the image file
  // @param const ColDescriptor &col - contains tensor implementation and datatype
  // @param std::shared_ptr<Tensor> tensor - return
  // @return Status The status code returned
  Status ReadImageToTensor(const std::string &path, const ColDescriptor &col, std::shared_ptr<Tensor> *tensor);

  // @param const std::string &path - path to the image file
  // @param TensorRow *row - return
  // @return Status The status code returned
  Status ReadAnnotationToTensor(const std::string &path, TensorRow *row);

  // Read image list from ImageSets
  // @return Status The status code returned
  Status ParseImageIds();

  // Read annotation from Annotation folder
  // @return Status The status code returned
  Status ParseAnnotationIds();

  // @param const std::string &path - path to annotation xml
  // @return Status The status code returned
  Status ParseAnnotationBbox(const std::string &path);

  // @param xmin - the left coordinate of bndbox
  // @param ymin - the top coordinate of bndbox
  // @param xmax - the right coordinate of bndbox
  // @param ymax - the bottom coordinate of bndbox
  // @param path - the file path of bndbox xml
  // @return Status The status code returned
  Status CheckIfBboxValid(const float &xmin, const float &ymin, const float &xmax, const float &ymax,
                          const std::string &path);

  // @param XMLElement *bbox_node - bbox node info found in json object
  // @param const char *name - sub node name in object
  // @param float *value - value of certain sub node
  // @return Status The status code returned
  void ParseNodeValue(XMLElement *bbox_node, const char *name, float *value);

  // Called first when function is called
  // @return Status The status code returned
  Status LaunchThreadsAndInitOp() override;

  // Private function for computing the assignment of the column name map.
  // @return - Status
  Status ComputeColMap() override;

  bool decode_;
  int64_t row_cnt_;
  std::string folder_path_;
  TaskType task_type_;
  std::string usage_;
  std::unique_ptr<DataSchema> data_schema_;
  bool extra_metadata_;

  std::vector<std::string> image_ids_;
  std::map<std::string, int32_t> class_index_;
  std::map<std::string, int32_t> label_index_;
  std::map<std::string, Annotation> annotation_map_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_VOC_OP_H_
