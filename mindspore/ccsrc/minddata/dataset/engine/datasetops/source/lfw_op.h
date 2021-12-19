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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_LFW_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_LFW_OP_H_

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
/// \class LFWOp
/// \brief A source dataset for reading and parsing LFW dataset.
class LFWOp : public MappableLeafOp {
 public:
  // Constructor
  // @param int32_t num_workers - num of workers reading images in parallel.
  // @param std::string dataset_dir - dir directory of LFW.
  // @param std::string task - set the task type of reading LFW.
  // @param std::string usage - split of LFW.
  // @param std::string image_set - image set of image funneling to use.
  // @param int32_t queue_size - connector queue size.
  // @param bool decode - decode the images after reading.
  // @param std::unique_ptr<dataschema> data_schema - schema of data.
  // @param td::unique_ptr<Sampler> sampler - sampler tells LFWOp what to read.
  LFWOp(int32_t num_workers, const std::string &dataset_dir, const std::string &task, const std::string &usage,
        const std::string &image_set, int32_t queue_size, bool decode, std::unique_ptr<DataSchema>,
        const std::shared_ptr<SamplerRT> &sampler);

  // Destructor.
  ~LFWOp() = default;

  // A print method typically used for debugging.
  // @param out The output stream to write output to.
  // @param show_all A bool to control if you want to show all info or just a summary.
  void Print(std::ostream &out, bool show_all) const override;

  // Function to count the number of samples in the LFWDataset.
  // @param int64_t *count - output rows number of LFWDataset.
  // @return Status The status code returned.
  Status CountTotalRows(int64_t *count);

  // Op name getter.
  // @return Name of the current Op.
  std::string Name() const override { return "LFWOp"; }

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

  // Read txt file.
  // @param const std::string &annotation_file_path - path to the txt file.
  // @return std::vector<std::vector<std::string>> The txt line context.
  std::vector<std::vector<std::string>> ReadFile(const std::string &annotation_file_path) const;

  // Read image list from ImageSets.
  // @return Status The status code returned.
  Status ParseImageIds();

  // Read image list from ImageSets when task is people.
  // @param const std::vector<std::vector<std::string>> &annotation_vector_string - contains the information
  //     of each line read from the txt file.
  // @return Status The status code returned.
  Status ParsePeopleImageIds(const std::vector<std::vector<std::string>> &annotation_vector_string);

  // Read image list from ImageSets when task is pairs.
  // @param const std::vector<std::vector<std::string>> &annotation_vector_string - contains the information
  //     of each line read from the txt file.
  // @return Status The status code returned.
  Status ParsePairsImageIds(const std::vector<std::vector<std::string>> &annotation_vector_string);

  // Gets the class indexing.
  // @return Status The status code returned.
  Status GetClassIndexing();

  // Private function for computing the assignment of the column name map.
  // @return Status The status code returned.
  Status ComputeColMap() override;

  // Parse LFW dataset files.
  // @return Status The status code returned.
  Status PrepareData() override;

 private:
  bool decode_;
  int64_t row_cnt_;
  std::string folder_path_;
  std::string task_;
  std::string usage_;
  std::string image_set_;
  std::unique_ptr<DataSchema> data_schema_;
  std::vector<std::vector<std::string>> image_label_people_;
  std::vector<std::vector<std::string>> image_label_pair_;
  std::map<std::string, uint32_t> class_index_;
  std::string real_folder_path_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_LFW_OP_H_
