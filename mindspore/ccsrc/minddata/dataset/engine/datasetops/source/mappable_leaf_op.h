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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_MAPPABLE_LEAF_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_MAPPABLE_LEAF_OP_H_

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

class MappableLeafOp : public ParallelOp, public RandomAccessOp {
 public:
  // Constructor
  // @param int32_t num_wkrs - Num of workers reading images in parallel
  // @param int32_t - rows_per_buffer Number of images (rows) in each buffer
  // @param std::string - dir directory of ImageNetFolder
  // @param int32_t queue_size - connector queue size
  // @param std::set<std::string> exts - set of file extensions to read, if empty, read everything under the dir
  // @param td::unique_ptr<Sampler> sampler - sampler tells the source what to read
  MappableLeafOp(int32_t num_wkrs, int32_t queue_size, std::shared_ptr<SamplerRT> sampler, int32_t rows_per_buffer);

  // Destructor.
  ~MappableLeafOp() = default;

  // Main Loop of MappableLeaf
  // Master thread: Fill IOBlockQueue, then goes to sleep
  // Worker thread: pulls IOBlock from IOBlockQueue, work on it then put buffer to mOutConnector
  // @return Status The status code returned
  Status operator()() override;

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return "MappableLeafPp"; }

 protected:
  // Initialize Sampler, calls sampler->Init() within
  // @return Status The status code returned
  Status InitSampler();

  //  // Called first when function is called
  //  // @return
  virtual Status LaunchThreadsAndInitOp() = 0;

  Status WorkerEntry(int32_t workerId) override;

  // @param const std::vector<int64_t> &keys - keys in ioblock
  // @param std::unique_ptr<DataBuffer> db
  // @return Status The status code returned
  Status LoadBuffer(const std::vector<int64_t> &keys, std::unique_ptr<DataBuffer> *db);

  // Load a tensor row according to a pair
  // @param row_id_type row_id - id for this tensor row
  // @param ImageLabelPair pair - <imagefile,label>
  // @param TensorRow row - loaded row
  // @return Status The status code returned
  virtual Status LoadTensorRow(row_id_type row_id, TensorRow *row) = 0;

  // reset Op
  // @return Status The status code returned
  Status Reset() override;

  int32_t rows_per_buffer_;
  int64_t row_cnt_;
  int64_t buf_cnt_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_MAPPABLE_LEAF_OP_H_
