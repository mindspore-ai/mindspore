/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef DATASET_ENGINE_DATASETOPS_EPOCH_CTRL_OP_H_
#define DATASET_ENGINE_DATASETOPS_EPOCH_CTRL_OP_H_

#include <memory>
#include <string>
#include <vector>
#include "minddata/dataset/engine/datasetops/repeat_op.h"
#include "minddata/dataset/engine/datasetops/pipeline_op.h"

namespace mindspore {
namespace dataset {
class EpochCtrlOp : public RepeatOp {
 public:
  class Builder : public RepeatOp::Builder {
   public:
    // Builder constructor.  Creates the builder object.
    // @note No default args
    // @param count - The number of repeats to do
    // @return This is a constructor.
    explicit Builder(int32_t count) : RepeatOp::Builder(count) {}

    // Default destructor
    ~Builder() = default;

    // The builder "build" method creates the final object.
    // @return shared_ptr to the new EpochCtrlOp object
    Status Build(std::shared_ptr<EpochCtrlOp> *);
  };

  // Constructor
  explicit EpochCtrlOp(int32_t num_epoch);

  // Destructor
  ~EpochCtrlOp();

  // A print method typically used for debugging
  // @param out - The output stream to write output to
  // @param show_all - A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override;
  std::string Name() const override { return kEpochCtrlOp; }

  // This function returns the buffer that is at the top of our output connector. The caller is
  // typically our parent node, when the parent is asking us to provide the next buffer of data.
  // Since EpochCtrlOp is derived from RepeatOp which is an inlined op, getting a buffer from us
  // will simply bounce you to get a buffer from our child.
  // Epoch Control Op does not eat the EOE, it will pass the EOE to the next op.
  Status GetNextBuffer(std::unique_ptr<DataBuffer> *p_buffer, int32_t worker_id, bool retry_if_eoe) override;

  // Base-class override for handling cases when an eoe is received.
  // @param worker_id - The worker id
  Status EoeReceived(int32_t worker_id) override;

  int64_t GetTreeRepeatCount() override;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_DATASETOPS_EPOCH_CTRL_OP_H_
