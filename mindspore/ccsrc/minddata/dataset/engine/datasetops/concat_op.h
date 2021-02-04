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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_CONCAT_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_CONCAT_OP_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>
#include "minddata/dataset/engine/datasetops/pipeline_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/distributed_sampler.h"

namespace mindspore {
namespace dataset {
class ConcatOp : public PipelineOp {
 public:
  // The nested builder class inside of the ConcatOp is used to help manage all of the arguments
  // for constructing it.  This Concat op is very simple though, so this builder is really just
  // provided for a consistent look and feel for creators of Dataset operators overall.
  class Builder {
   public:
    // Builder constructor. Creates the builder object.
    // @note No default args
    // @return This is a constructor.
    Builder();

    // Default destructor
    ~Builder() = default;

    // The builder "build" method creates the final object.
    // @return shared_ptr to the new ConcatOp object
    Status Build(std::shared_ptr<ConcatOp> *);
    Builder &SetSampler(std::shared_ptr<SamplerRT> sampler) {
      builder_sampler_ = std::move(sampler);
      return *this;
    }

    Builder &SetChildrenFlagAndNums(std::vector<std::pair<int, int>> children_flag_and_nums) {
      children_flag_and_nums_ = std::move(children_flag_and_nums);
      return *this;
    }

    Builder &SetChildrenStartEndIndex(std::vector<std::pair<int, int>> children_start_end_index) {
      children_start_end_index_ = std::move(children_start_end_index);
      return *this;
    }

   private:
    int32_t builder_op_connector_size_;
    std::shared_ptr<SamplerRT> builder_sampler_;
    std::vector<std::pair<int, int>> children_flag_and_nums_;
    std::vector<std::pair<int, int>> children_start_end_index_;
  };

  // Constructor of the ConcatOp.
  // @note The builder class should be used to call it
  // @param op_connector_size - connector size
  explicit ConcatOp(int32_t op_connector_size);
  ConcatOp(int32_t op_connector_size, const std::shared_ptr<SamplerRT> &sampler,
           const std::vector<std::pair<int, int>> &children_flag_and_nums,
           const std::vector<std::pair<int, int>> &children_start_end_index);

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

 private:
  Status Verify(int32_t id, const std::unique_ptr<DataBuffer> &buf);

  int32_t children_num_;                                     // The num of child of parent node.
  std::unordered_map<std::string, int32_t> column_name_id_;  // Mapping between col index and col name
  std::vector<DataType> data_type_;
  std::vector<dsize_t> data_rank_;
  std::shared_ptr<SamplerRT> sampler_;
  std::vector<std::pair<int, int>> children_flag_and_nums_;
  std::vector<std::pair<int, int>> children_start_end_index_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_CONCAT_OP_H_
