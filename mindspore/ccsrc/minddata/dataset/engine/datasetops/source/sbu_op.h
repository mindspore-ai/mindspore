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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SBU_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SBU_OP_H_

#include <algorithm>
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
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/wait_post.h"

namespace mindspore {
namespace dataset {

using SBUImageCaptionPair = std::pair<Path, std::string>;

class SBUOp : public MappableLeafOp {
 public:
  // Constructor.
  // @param const std::string &folder_path - dir directory of SBU data file.
  // @param bool decode - whether to decode images.
  // @param std::unique_ptr<DataSchema> data_schema - the schema of the SBU dataset.
  // @param std::unique_ptr<Sampler> sampler - sampler tells SBUOp what to read.
  // @param int32_t num_workers - number of workers reading images in parallel.
  // @param int32_t queue_size - connector queue size.
  SBUOp(const std::string &folder_path, bool decode, std::unique_ptr<DataSchema> data_schema,
        std::shared_ptr<SamplerRT> sampler, int32_t num_workers, int32_t queue_size);

  // Destructor.
  ~SBUOp() = default;

  // Op name getter.
  // @return std::string - Name of the current Op.
  std::string Name() const override { return "SBUOp"; }

  // A print method typically used for debugging.
  // @param std::ostream &out - out stream.
  // @param bool show_all - whether to show all information.
  void Print(std::ostream &out, bool show_all) const override;

  // Function to count the number of samples in the SBU dataset.
  // @param const std::string &dir - path to the SBU directory.
  // @param int64_t *count - output arg that will hold the minimum of the actual dataset size and numSamples.
  // @return Status - The status code returned.
  static Status CountTotalRows(const std::string &dir, int64_t *count);

 private:
  // Load a tensor row according to a pair.
  // @param row_id_type row_id - id for this tensor row.
  // @param TensorRow row - image & label read into this tensor row.
  // @return Status - The status code returned.
  Status LoadTensorRow(row_id_type row_id, TensorRow *row) override;

  // Private function for computing the assignment of the column name map.
  // @return Status - The status code returned.
  Status ComputeColMap() override;

  // Called first when function is called.
  // @return Status - The status code returned.
  Status LaunchThreadsAndInitOp() override;

  // @param const std::string &path - path to the image file.
  // @param std::shared_ptr<Tensor> tensor - tensor to store image.
  // @return Status - The status code returned.
  Status ReadImageToTensor(const std::string &path, std::shared_ptr<Tensor> *tensor);

  // Parse SBU data file.
  // @return Status - The status code returned.
  Status ParseSBUData();

  // Get available image-caption pairs.
  // @param std::ifstream &url_file_reader - url file reader.
  // @param std::ifstream &caption_file_reader - caption file reader.
  // @return Status - The status code returned.
  Status GetAvailablePairs(std::ifstream &url_file_reader, std::ifstream &caption_file_reader);

  // Parse path-caption pair.
  // @param const std::string &url - image url.
  // @param const std::string &caption - caption.
  // @return Status - The status code returned.
  Status ParsePair(const std::string &url, const std::string &caption);

  // A util for string replace.
  // @param std::string *str - string to be replaces.
  // @param const std::string &from - string from.
  // @param const std::string &to - string to.
  // @return Status - The status code returned.
  Status ReplaceAll(std::string *str, const std::string &from, const std::string &to);

  std::string folder_path_;  // directory of data files
  const bool decode_;
  std::unique_ptr<DataSchema> data_schema_;

  Path url_path_;
  Path caption_path_;
  Path image_folder_;
  std::vector<SBUImageCaptionPair> image_caption_pairs_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SBU_OP_H_
