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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_CONCAT_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_CONCAT_OP_H_

#include <memory>
#include <string>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>
#include "minddata/dataset/engine/dataset_iterator.h"
#include "minddata/dataset/engine/datasetops/pipeline_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/distributed_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/random_sampler.h"

namespace mindspore {
namespace dataset {
class ConcatOp : public PipelineOp {
 public:
  // Constructor of the ConcatOp.
  // @note The builder class should be used to call it
  // @param op_connector_size - connector size
  ConcatOp();
  ConcatOp(const std::shared_ptr<SamplerRT> &sampler, const std::vector<std::pair<int, int>> &children_flag_and_nums,
           const std::vector<std::pair<int, int>> &children_start_end_index,
           const std::vector<int64_t> &children_sizes);

  // Destructor
  ~ConcatOp() = default;

  // A print method typically used for debugging
  // @param out - The output stream to write output to
  // @param show_all - A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override;

  // << Stream output operator overload
  // @notes This allows you to write the debug print info using stream operators
  // @param out - reference to the output stream being overloaded
  // @param ro - reference to the ConcatOp to display
  // @return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const ConcatOp &ro) {
    ro.Print(out, false);
    return out;
  }

  // All dataset ops operate by launching a thread (see ExecutionTree). This class functor will
  // provide the master loop that drives the logic for performing the work
  // @return Status The status code returned
  Status operator()() override;

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return kConcatOp; }

  // Private function for computing the assignment of the column name map.
  // @return - Status
  Status ComputeColMap() override;

  /// \brief Gets the number of classes
  /// \param[out] num_classes the number of classes
  /// \return Status - The status code return
  Status GetNumClasses(int64_t *num_classes) override;

  Status GetNextRow(TensorRow *row) override;

  Status GetNextRowPullMode(TensorRow *const row) override;

  Status SampleInSequence(TensorRow *row, bool is_pipeline_mode = true);

  Status SampleInGlobal(TensorRow *row, bool is_pipeline_mode = true);

  /// Check if the current sample will be taken or dropped
  /// \return bool
  bool IgnoreSample();

 protected:
  /// \brief Gets the implementation status for operator in pull mode
  /// \return implementation status
  ImplementedPullMode PullModeImplementationStatus() const override { return ImplementedPullMode::Implemented; }

 private:
  Status Verify(int32_t id, const TensorRow &new_row);

  std::unordered_map<std::string, int32_t> column_name_id_;  // Mapping between col index and col name
  std::vector<DataType> data_type_;
  std::vector<dsize_t> data_rank_;
  std::vector<std::pair<int, int>> children_flag_and_nums_;
  std::vector<std::pair<int, int>> children_start_end_index_;
  std::vector<int64_t> children_sizes_;
  std::vector<int64_t> children_sizes_ori_;
  std::vector<bool> children_exhausted_;

  size_t cur_child_;
  bool verified_;
  int64_t sample_number_;

  int32_t num_shard_;
  int32_t shard_index_;

  std::unique_ptr<std::discrete_distribution<>> discrete_random_;
  bool global_shuffle_;
  uint32_t seed_;
  std::mt19937 rnd_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_CONCAT_OP_H_
