/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/dataset_iterator.h"

namespace mindspore {
namespace dataset {
class SkipOp : public PipelineOp {
 public:
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
  // All dataset ops operate by launching a thread (see ExecutionTree). This class functor will
  // provide the master loop that drives the logic for performing the work
  // @return Status The status code returned
  Status operator()() override;

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return kSkipOp; }
  Status GetNextRow(TensorRow *row) override;

  void SetOnceOnly(bool once_only) { once_only_ = once_only; }

  /// \brief Gets the next row
  /// \param row[out] - Fetched TensorRow
  /// \return Status The status code returned
  Status GetNextRowPullMode(TensorRow *const row) override;

 protected:
  /// \brief Gets the implementation status for operator in pull mode
  /// \return implementation status
  ImplementedPullMode PullModeImplementationStatus() const override { return ImplementedPullMode::Implemented; }

 private:
  int32_t max_skips_;   // The number of skips that the user requested
  int32_t skip_count_;  // A counter for the current number of executed skips

  bool once_only_ = false;     // skip for skip_count_ steps only once
  int64_t data_produced_ = 0;  // The number of data has been pushed to the next op

  std::unique_ptr<ChildIterator> child_iterator_;  // An iterator for fetching.
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SKIP_OP_H_
