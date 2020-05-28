/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef DATASET_ENGINE_DATASETOPS_REPEAT_OP_H_
#define DATASET_ENGINE_DATASETOPS_REPEAT_OP_H_

#include <memory>
#include <string>
#include <vector>
#include "dataset/engine/datasetops/pipeline_op.h"

namespace mindspore {
namespace dataset {
class RepeatOp : public PipelineOp {
 public:
  static constexpr int32_t kInfiniteRepeat = -1;

  // The nested builder class inside of the RepeatOp is used to help manage all of the arguments
  // for constructing it.  This repeat op is very simple though, so this builder is really just
  // provided for a consistent look and feel for creators of Dataset operators overall.
  class Builder {
   public:
    // Builder constructor.  Creates the builder object.
    // @note No default args
    // @param count - The number of repeats to do
    // @return This is a constructor.
    explicit Builder(int32_t count);

    // Default destructor
    ~Builder() = default;

    // The builder "build" method creates the final object.
    // @return shared_ptr to the new StorageOp object
    Status Build(std::shared_ptr<RepeatOp> *);

   private:
    int32_t build_max_repeats_;

    Status SanityCheck() const;
  };

  // Constructor of the RepeatOp.
  // @note The builder class should be used to call it
  // @param count - The number of repeats to do
  explicit RepeatOp(int32_t count);

  // Destructor
  ~RepeatOp();

  // A print method typically used for debugging
  // @param out - The output stream to write output to
  // @param show_all - A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override;

  // << Stream output operator overload
  // @notes This allows you to write the debug print info using stream operators
  // @param out - reference to the output stream being overloaded
  // @param ro - reference to the RepeatOp to display
  // @return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const RepeatOp &ro) {
    ro.Print(out, false);
    return out;
  }

  // Class functor operator () override.
  // Most dataset ops operate by launching a thread (see ExecutionTree).
  // However, the RepeatOp is defined as a inlined operator, so it is invalid to launch the
  // functor since this op runs inlined inside another operator.  The function is overloaded to
  // ensure that it is not called by mistake (it will generate an error).
  // @return Status - The error code return
  Status operator()() override;

  // Base-class override for setting specific RepeatOp configurations. This code will be called
  // during the execution tree prepare phase BEFORE traversing down to child operators.
  uint32_t PrepareFlags() const override;

  // Base-class override for executing specific RepeatOp configurations. This code will be called
  // during the execution tree post-prepare phase when it is visiting this operator.
  Status PrepareNodePostAction() override;

  // This function returns the buffer that is at the top of our output connector. The caller is
  // typically our parent node, when the parent is asking us to provide the next buffer of data.
  // Since RepeatOp is an inlined op, getting a buffer from us will simply bounce you to get
  // a buffer from our child.
  // @note This function sets the `retryIfEoe` flag when popping from the child connector. This way,
  // this function will retry to pop the connector again and will get the non-EOE buffer if any.
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

  // Base-class override. Return the number of workers in the first parent.
  // @param workerId - The worker id
  int32_t num_consumers() const override;

  // Base-class override. Return the number of producers in the first child.
  // @param workerId - The worker id
  int32_t num_producers() const override;

  // Base-class override for NodePass visitor acceptor.
  // @param p - Pointer to the NodePass to be accepted.
  // @param modified - Whether this node visit modified the pipeline.
  // @return - Status of the node visit.
  Status Accept(NodePass *p, bool *modified) override;

 private:
  int32_t max_repeats_;                              // The number of repeats that the user requested
  int32_t repeat_count_;                             // A counter for the current number of executed repeats
  std::vector<std::shared_ptr<DatasetOp>> eoe_ops_;  // List of operators that can generate EOE underneath this repeat.
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_DATASETOPS_REPEAT_OP_H_
