/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_PROJECT_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_PROJECT_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/datasetops/pipeline_op.h"

namespace mindspore {
namespace dataset {
class ProjectOp : public PipelineOp {
 public:
  // Constructor of the ProjectOp.
  // @param columnsToProject -
  explicit ProjectOp(const std::vector<std::string> &columns_to_project);

  // Destructor.
  ~ProjectOp() = default;

  // A print method typically used for debugging.
  // @param out - The output stream to write output to.
  // @param show_all - A bool to control if you want to show all info or just a summary.
  void Print(std::ostream &out, bool show_all) const override;

  // << Stream output operator overload.
  // @notes This allows you to write the debug print info using stream operators.
  // @param out - reference to the output stream being overloaded.
  // @param project_op - reference to the ProjectOp to display.
  // @return - the output stream must be returned.
  friend std::ostream &operator<<(std::ostream &out, const ProjectOp &project_op) {
    project_op.Print(out, false);
    return out;
  }

  // Class functor operator () override.
  // Most dataset ops operate by launching a thread (see ExecutionTree).
  // However, the ProjectOp is defined as a inlined operator, so it is invalid to launch the
  // functor since this op runs inlined inside another operator. The function is overloaded to
  // ensure that it is not called by mistake (it will generate an error).
  // @return Status The status code returned
  Status operator()() override;

  // Gets a row from the child node and projects that row. The caller is typically our parent node.
  // @param row - output pointer to the projected row.
  // @param worker_id - The worker id
  Status GetNextRow(TensorRow *row) override;

  // Base-class override for special eoe handler.
  // Inline operators must override this because there is no connector to push eoe onto.
  // @return Status The status code returned
  Status EoeReceived(int32_t worker_id) override;

  // Base-class override for special eof handler.
  // Inline operators must override this because there is no connector to push eof onto.
  // @return Status The status code returned
  Status EofReceived(int32_t worker_id) override;

  /// \brief Gets the next row
  /// \param row[out] - Fetched TensorRow
  /// \return Status The status code returned
  Status GetNextRowPullMode(TensorRow *const row) override;

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return kProjectOp; }

 protected:
  /// \brief Gets the implementation status for operator in pull mode
  /// \return implementation status
  ImplementedPullMode PullModeImplementationStatus() const override { return ImplementedPullMode::Implemented; }

 private:
  std::vector<std::string> columns_to_project_;
  std::vector<int32_t> projected_column_indices_;

  TensorRow Project(const TensorRow &row);

  // Computing the assignment of the column name map.
  // @return - Status
  Status ComputeColMap() override;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_PROJECT_OP_H_
