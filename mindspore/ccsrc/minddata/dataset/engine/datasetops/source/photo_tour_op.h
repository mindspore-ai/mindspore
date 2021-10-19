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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_PHOTO_TOUR_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_PHOTO_TOUR_OP_H_

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

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
// Forward declares
template <typename T>
class Queue;

using MatchTuple = std::tuple<uint32_t, uint32_t, uint32_t>;

class PhotoTourOp : public MappableLeafOp {
 public:
  // Constructor
  // @param const std::string &datasetDir - Path to the root directory that
  //     contains the dataset.
  // @param const std::string &name - Name of the dataset to load.
  // @param const std::string &usage - 'train' or 'test', If 'train', the
  //     generated dataset has one column ["image"], else three columns
  //     ["image1", "image2", "matches"].
  // @param int32_t num_workers - number of workers reading images in parallel.
  // @param int32_t queue_size - connector queue size.
  // @param std::unique_ptr<DataSchema> data_schema - the schema of the photo tour dataset.
  // @param std::unique_ptr<Sampler> sampler - sampler tells PhotoTourOp what to read.
  PhotoTourOp(const std::string &dataset_dir, const std::string &name, const std::string &usage, int32_t num_workers,
              int32_t queue_size, std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler);

  // Destructor.
  ~PhotoTourOp() = default;

  // Method derived from RandomAccess Op, enable Sampler to get all ids for each class.
  // @param std::map<int32_t, std::vector<int64_t >> *cls_ids - key label, val all ids for this class.
  // @return Status - The status code returned.
  Status GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const override;

  // A print method typically used for debugging.
  // @param std::ostream &out - out stream.
  // @param bool show_all - whether to show all information.
  void Print(std::ostream &out, bool show_all) const override;

  // Function to count the number of samples in the PhotoTour dataset.
  // @param const std::string &dir - path to the PhotoTour directory.
  // @param const std::string &name - name of the dataset to load.
  // @param const std::string &usage - 'train' or 'test', If 'train', the
  //     generated dataset has one column ["image"], else three columns
  //     ["image1", "image2", "matches"].
  // @param int64_t *count - output arg that will hold the minimum of the actual dataset
  //     size and numSamples.
  // @return Status - The status code returned.
  static Status CountTotalRows(const std::string &dir, const std::string &name, const std::string &usage,
                               int64_t *count);

  // Op name getter.
  // @return std::string - Name of the current Op.
  std::string Name() const override { return "PhotoTourOp"; }

 private:
  // Load a tensor row according to the row_id.
  // @param row_id_type row_id - id for this tensor row.
  // @param TensorRow *row - load one piece of data into this tensor row.
  // @return Status - The status code returned.
  Status LoadTensorRow(row_id_type row_id, TensorRow *row);

  // Judge whether string s ends with string sub.
  // @param const std::string &s - full string.
  // @param const std::string &sub - suffix.
  // @return bool The Result of whether string s ends with string sub.
  bool EndsWith(const std::string &s, const std::string &sub);

  // Read the content in the given file path.
  // @param const std::string &info_file - info file name.
  // @param std::string *ans - store the content of the info file.
  // @return Status - The status code returned.
  Status GetFileContent(const std::string &info_file, std::string *ans);

  // Read the meta info for each patch.
  // @param const std::string &data_dir - data_dir stores the info file.
  // @param const std::string &info_file - info file name.
  // @return Status - The status code returned.
  Status ReadInfoFile(const std::string &data_dir, const std::string &info_file);

  // Read the matches meta info.
  // @param const std::string &data_dir - data_dir stores the info file.
  // @param const std::string &matches_file - matches info file name.
  // @return Status - The status code returned.
  Status ReadMatchedFile(const std::string &data_dir, const std::string &matches_file);

  // Get one piece of PhotoTour data.
  // @param uint32_t index - index of data to read.
  // @param std::shared_ptr<Tensor> *image_tensor - store the indexed data.
  // @return Status - The status code returned.
  Status GetPhotoTourDataTensor(uint32_t index, std::shared_ptr<Tensor> *image_tensor);

  // Read all files in the directory.
  // @return Status - The status code returned.
  Status PrepareData();

  // Private function for computing the assignment of the column name map.
  // @return Status - The status code returned.
  Status ComputeColMap() override;

  int64_t buf_cnt_;
  std::unique_ptr<DataSchema> data_schema_;

  const std::string dataset_dir_;           // directory of image folder
  const std::string name_;                  // dataset name
  const std::string usage_;                 // 'train' or 'test'
  std::string chosen_dataset_folder_path_;  // dataset_dir + name : folder
  bool train_;                              // whether the usage_ is "train" or not

  std::vector<std::string> image_names_;
  std::vector<cv::Mat> image_bmps_;

  std::vector<MatchTuple> matches_;  // train_ = false, stores the triplets (img1, img2, is_match)
  std::vector<uint32_t> labels_;     // label of i_th patch
  std::mutex access_mutex_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_PHOTO_TOUR_OP_H_
