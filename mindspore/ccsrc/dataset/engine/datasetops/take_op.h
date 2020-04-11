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
#ifndef DATASET_ENGINE_DATASETOPS_TAKE_OP_H_
#define DATASET_ENGINE_DATASETOPS_TAKE_OP_H_

#include <memory>
#include <string>
#include <vector>
#include "dataset/engine/datasetops/pipeline_op.h"

namespace mindspore {
namespace dataset {
class TakeOp : public PipelineOp {
 public:
  // The nested builder class inside of the TakeOp is used to help manage all of the arguments
  // for constructing it.  This take op is very simple though, so this builder is really just
  // provided for a consistent look and feel for creators of Dataset operators overall.
  class Builder {
   public:
    // Builder constructor.  Creates the builder object.
    // @note No default args
    // @param count - The number of takes to do
    // @return This is a constructor.
    explicit Builder(int32_t count);

    // Default destructor
    ~Builder() = default;

    // The builder "build" method creates the final object.
    // @return shared_ptr to the new StorageOp object
    Status Build(std::shared_ptr<TakeOp> *);

   private:
    int32_t build_max_takes_;

    Status SanityCheck() const;
  };

  // Constructor of the TakeOp.
  // @note The builder class should be used to call it
  // @param count - The number of takes to do
  explicit TakeOp(int32_t count);

  // Destructor
  ~TakeOp() = default;

  // A print method typically used for debugging
  // @param out - The output stream to write output to
  // @param show_all - A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override;

  // << Stream output operator overload
  // @notes This allows you to write the debug print info using stream operators
  // @param out - reference to the output stream being overloaded
  // @param ro - reference to the TakeOp to display
  // @return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const TakeOp &ro) {
    ro.Print(out, false);
    return out;
  }

  // Class functor operator () override.
  // Most dataset ops operate by launching a thread (see ExecutionTree).
  // However, the TakeOp is defined as a inlined operator, so it is invalid to launch the
  // functor since this op runs inlined inside another operator.  The function is overloaded to
  // ensure that it is not called by mistake (it will generate an error).
  // @return Status - The error code return
  Status operator()() override;

  // Gets a buffer from the child node. The caller is typically our parent node.
  // @note This function sets the `retryIfEoe` flag when popping from the child connector. This way,
  // this function will retry to pop the connector again and will get the non-EOE buffer if any.
  // @param p_buffer - output pointer to the buffer that it will fetch.
  // @param worker_id - The worker id
  // @param retry_if_eoe Set this flag to true to allow calling pop() again after the first pop() returns EOE.
  // @return Status - The error code return
  Status GetNextBuffer(std::unique_ptr<DataBuffer> *p_buffer, int32_t worker_id, bool retry_if_eoe) override;

  // During tree prepare phase, operators may have specific post-operations to perform depending on
  // their role.
  // @notes Derived versions of this function should always call it's superclass version first
  // before providing their own implementations.
  Status PrepareNodePostAction() override;

 private:
  int32_t max_takes_;   // The number of takes that the user requested
  int32_t take_count_;  // A counter for the current number of executed takes

  Status FillBuffer(std::unique_ptr<DataBuffer> *buffer, std::unique_ptr<DataBuffer> *data_buffer);
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_DATASETOPS_TAKE_OP_H_
