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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_TAKE_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_TAKE_OP_H_

#include <memory>
#include <string>
#include <vector>
#include "minddata/dataset/engine/datasetops/pipeline_op.h"

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
    // @return shared_ptr to the new TakeOp object
    Status Build(std::shared_ptr<TakeOp> *);

   private:
    int32_t build_max_takes_;
    int32_t builder_op_connector_size_;

    Status SanityCheck() const;
  };

  // Constructor of the TakeOp.
  // @note The builder class should be used to call it
  // @param count - The number of takes to do
  explicit TakeOp(int32_t count, int32_t op_connector_size);

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

  // All dataset ops operate by launching a thread (see ExecutionTree). This class functor will
  // provide the master loop that drives the logic for performing the work
  // @return Status The status code returned
  Status operator()() override;

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return kTakeOp; }

 private:
  int32_t max_takes_;   // The number of takes that the user requested
  int32_t take_count_;  // A counter for the current number of executed takes

  Status FillBuffer(std::unique_ptr<DataBuffer> *buffer, std::unique_ptr<DataBuffer> *data_buffer);
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_TAKE_OP_H_
