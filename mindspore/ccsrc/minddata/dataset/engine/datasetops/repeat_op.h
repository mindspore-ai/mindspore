/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_REPEAT_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_REPEAT_OP_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "minddata/dataset/engine/datasetops/pipeline_op.h"

namespace mindspore {
namespace dataset {
class RepeatOp : public PipelineOp {
 public:
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
  // @return Status The status code returned
  Status operator()() override;

  // This function returns the row that is at the top of our output connector. The caller is
  // typically our parent node, when the parent is asking us to provide the next row of data.
  // Since RepeatOp is an inlined op, getting a row from us will simply bounce you to get
  // a row from our child.
  // @param row - output pointer to the buffer that it will fetch.
  // @return Status The status code returned
  Status GetNextRow(TensorRow *row) override;

  // Base-class override for handling cases when an eoe is received.
  // @param worker_id - The worker id
  Status EoeReceived(int32_t worker_id) override;

  // Base-class override for handling cases when an eof is received.
  // @param worker_id - The worker id
  Status EofReceived(int32_t worker_id) override;

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return kRepeatOp; }

  /// \brief Getter function
  /// \return The number of repeats that the user requested
  int32_t num_repeats() { return num_repeats_; }

  /// \brief reset Op
  /// \@return Status The status code returned
  Status Reset() override;

  int64_t GetTreeRepeatCount() override;

  // \brief Adds an operator to the repeat ops list of tracked leaf/eoe nodes
  // \param[in] eoe_op The input leaf/eoe operator to add to the list
  void AddToEoeList(std::shared_ptr<DatasetOp> eoe_op) { eoe_ops_.push_back(std::move(eoe_op)); }

  std::vector<std::shared_ptr<DatasetOp>> eoe_ops_;  // List of operators that can generate EOE underneath this repeat.

  /// \brief In pull mode, gets the next row
  /// \param row[out] - Fetched TensorRow
  /// \return Status The status code returned
  Status GetNextRowPullMode(TensorRow *const row) override;

 protected:
  // The number of repeats that the user requested.
  // Note that num_repeats_ is different with op_total_repeats_ or op_num_repeats_per_epoch_ in base DatasetOp class.
  // For example, for repeat1 op in pipeline tfreader -> repeat1(3) -> repeat2(2) -> epoch ctrl(4),
  // num_repeats_ = 3, op_total_repeats_ = 24, op_num_repeats_per_epoch_ = 6.
  int32_t num_repeats_;
  // A counter for the current number of executed repeats.
  // Note that repeat_count_ is different with op_current_repeats_ in the base DatasetOp class
  // because it counts the repeats in the current epoch, whereas op_current_repeats_ counts the global total repeats.
  int32_t repeat_count_;

  /// \brief Gets the implementation status for operator in pull mode
  /// \return implementation status
  ImplementedPullMode PullModeImplementationStatus() const override { return ImplementedPullMode::Implemented; }
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_REPEAT_OP_H_
