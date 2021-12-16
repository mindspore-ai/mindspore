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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_KITTI_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_KITTI_OP_H_

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
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/wait_post.h"

namespace mindspore {
namespace dataset {
// Forward declares
template <typename T>
class Queue;

using Annotation = std::vector<std::pair<std::string, std::vector<float>>>;

/// \class KITTIOp
/// \brief A source dataset for reading and parsing KITTI dataset.
class KITTIOp : public MappableLeafOp {
 public:
  // Constructor
  // @param std::string dataset_dir - dir directory of KITTI.
  // @param std::string usage - split of KITTI.
  // @param int32_t num_workers - number of workers reading images in parallel.
  // @param int32_t queue_size - connector queue size.
  // @param bool decode - whether to decode images.
  // @param std::unique_ptr<DataSchema> data_schema - the schema of the KITTI dataset.
  // @param std::shared_ptr<Sampler> sampler - sampler tells KITTIOp what to read.
  KITTIOp(const std::string &dataset_dir, const std::string &usage, int32_t num_workers, int32_t queue_size,
          bool decode, std::unique_ptr<DataSchema> data_schema, const std::shared_ptr<SamplerRT> &sampler);

  // Destructor.
  ~KITTIOp() = default;

  // A print method typically used for debugging.
  // @param out - The output stream to write output to.
  // @param show_all - A bool to control if you want to show all info or just a summary.
  void Print(std::ostream &out, bool show_all) const override;

  // Function to count the number of samples in the KITTIDataset.
  // @param int64_t *count - output rows number of KITTIDataset.
  Status CountTotalRows(int64_t *count);

  // Op name getter.
  // @return Name of the current Op.
  std::string Name() const override { return "KITTIOp"; }

 private:
  // Load a tensor row according to image id.
  // @param row_id_type row_id - id for this tensor row.
  // @param TensorRow *row - image & target read into this tensor row.
  // @return Status The status code returned.
  Status LoadTensorRow(row_id_type row_id, TensorRow *row) override;

  // Load an image to Tensor.
  // @param const std::string &path - path to the image file.
  // @param const ColDescriptor &col - contains tensor implementation and datatype.
  // @param std::shared_ptr<Tensor> *tensor - return image tensor.
  // @return Status The status code returned.
  Status ReadImageToTensor(const std::string &path, const ColDescriptor &col, std::shared_ptr<Tensor> *tensor);

  // Load an annotation to Tensor.
  // @param const std::string &path - path to the image file.
  // @param TensorRow *row - return annotation tensor.
  // @return Status The status code returned.
  Status ReadAnnotationToTensor(const std::string &path, TensorRow *row);

  // Read image list from ImageSets.
  // @return Status The status code returned.
  Status ParseImageIds();

  // Read annotation from Annotation folder.
  // @return Status The status code returned.
  Status ParseAnnotationIds();

  // Function to parse annotation bbox.
  // @param const std::string &path - path to annotation xml.
  // @return Status The status code returned.
  Status ParseAnnotationBbox(const std::string &path);

  // Private function for computing the assignment of the column name map.
  // @return Status The status code returned.
  Status ComputeColMap() override;

 protected:
  Status PrepareData() override;

 private:
  bool decode_;
  int64_t row_cnt_;
  std::string folder_path_;
  std::string usage_;
  std::unique_ptr<DataSchema> data_schema_;
  std::vector<std::string> image_ids_;
  std::map<std::string, uint32_t> label_index_;
  std::map<std::string, Annotation> annotation_map_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_KITTI_OP_H_
