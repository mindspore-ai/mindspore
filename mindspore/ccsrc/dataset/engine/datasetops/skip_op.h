/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef DATASET_ENGINE_DATASETOPS_SKIP_OP_H_
#define DATASET_ENGINE_DATASETOPS_SKIP_OP_H_

#include <memory>
#include <string>
#include <vector>
#include "dataset/engine/datasetops/pipeline_op.h"

namespace mindspore {
namespace dataset {
class SkipOp : public PipelineOp {
 public:
  class Builder {
   public:
    // Builder constructor.  Creates the builder object.
    // @note No default args
    // @param count - The number of skip to do
    // @return This is a constructor.
    explicit Builder(int32_t count);

    // Default destructor
    ~Builder() = default;

    // The builder "build" method creates the final object.
    // @return shared_ptr to the new StorageOp object
    Status Build(std::shared_ptr<SkipOp> *);

   private:
    int32_t build_max_skips_;

    Status SanityCheck() const;
  };

  // Constructor of the SkipOp.
  // @note The builder class should be used to call it
  // @param count - The number of skips to do
  explicit SkipOp(int32_t count);

  // Destructor
  ~SkipOp();

  // A print method typically used for debugging
  // @param out - The output stream to write output to
  // @param show_all - A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override;

  // Class functor operator () override.
  // Most dataset ops operate by launching a thread (see ExecutionTree).
  // However, the SkipOp is defined as a inlined operator, so it is invalid to launch the
  // functor since this op runs inlined inside another operator.  The function is overloaded to
  // ensure that it is not called by mistake (it will generate an error).
  // @return Status - The error code return
  Status operator()() override;

  // This function returns the buffer that is at the top of our output connector. The caller is
  // typically our parent node, when the parent is asking us to provide the next buffer of data.
  // Since SkipOp is an inlined op, getting a buffer from us will simply bounce you to get
  // a buffer from our child.
  // @param p_buffer - output pointer to the buffer that it will fetch.
  // @param worker_id - The worker id
  // @param retry_if_eoe Set this flag to true to allow calling pop() again after the first pop() returns EOE.
  // @return Status - The error code return
  Status GetNextBuffer(std::unique_ptr<DataBuffer> *p_buffer, int32_t worker_id, bool retry_if_eoe) override;

  // Base-class override for handling cases when an eoe is received.
  // @param worker_id - The worker id
  Status EoeReceived(int32_t worker_id) override;

  // Base-class override for handling cases when an eof is received.
  // @param worker_id - The worker id
  Status EofReceived(int32_t worker_id) override;

 private:
  int32_t max_skips_;   // The number of skips that the user requested
  int32_t skip_count_;  // A counter for the current number of executed skips
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_DATASETOPS_SKIP_OP_H_
