/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/dataset_iterator.h"

namespace mindspore {
namespace dataset {
class TakeOp : public PipelineOp {
 public:
  // Constructor of the TakeOp.
  // @note The builder class should be used to call it
  // @param count - The number of takes to do
  explicit TakeOp(int32_t count);

  // Destructor
  ~TakeOp() = default;

  // A print method typically used for debugging
  // \param[in] out - The output stream to write output to
  // \param[in] show_all - A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override;

  // << Stream output operator overload
  // @notes This allows you to write the debug print info using stream operators
  // \param[in] out - reference to the output stream being overloaded
  // \param[in] ro - reference to the TakeOp to display
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

  /// \brief Gets the next row
  /// \param row[out] - Fetched TensorRow
  /// \return Status The status code returned
  Status GetNextRow(TensorRow *row) override;

  /// \brief In pull mode, gets the next row
  /// \param row[out] - Fetched TensorRow
  /// \return Status The status code returned
  Status GetNextRowPullMode(TensorRow *const row) override;

 protected:
  /// \brief Gets the implementation status for operator in pull mode
  /// \return implementation status
  ImplementedPullMode PullModeImplementationStatus() const override { return ImplementedPullMode::Implemented; }

 private:
  int32_t max_takes_;   // The number of takes that the user requested
  int32_t take_count_;  // A counter for the current number of executed takes

  std::unique_ptr<ChildIterator> child_iterator_;  // An iterator for fetching.

  /// \brief Common non-pull mode and pull mode function to get the next row
  /// \param row[out] - Fetched TensorRow
  /// \param is_pull_mode[in] - an indicator to identify if in pull mode or not
  /// \return Status The status code returned
  Status CommonGetNextRow(TensorRow *row, bool is_pull_mode);
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_TAKE_OP_H_
