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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SKIP_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SKIP_OP_H_

#include <memory>
#include <string>
#include <vector>
#include "minddata/dataset/engine/datasetops/pipeline_op.h"

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
    // @return shared_ptr to the new SkipOp object
    Status Build(std::shared_ptr<SkipOp> *);

   private:
    int32_t build_max_skips_;
    int32_t builder_op_connector_size_;

    Status SanityCheck() const;
  };

  // Constructor of the SkipOp.
  // @note The builder class should be used to call it
  // @param count - The number of skips to do
  explicit SkipOp(int32_t count, int32_t op_connector_size);

  // Destructor
  ~SkipOp();

  // A print method typically used for debugging
  // @param out - The output stream to write output to
  // @param show_all - A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override;

  // Class functor operator () override.
  // All dataset ops operate by launching a thread (see ExecutionTree). This class functor will
  // provide the master loop that drives the logic for performing the work
  // @return Status The status code returned
  Status operator()() override;

  // Base-class override for handling cases when an eoe is received.
  // @param worker_id - The worker id
  Status EoeReceived(int32_t worker_id) override;

  // Base-class override for handling cases when an eof is received.
  // @param worker_id - The worker id
  Status EofReceived(int32_t worker_id) override;

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return kSkipOp; }

 private:
  int32_t max_skips_;   // The number of skips that the user requested
  int32_t skip_count_;  // A counter for the current number of executed skips
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SKIP_OP_H_
