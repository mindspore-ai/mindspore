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
#ifndef DATASET_ENGINE_DATASETOPS_BUCKET_BATCH_BY_LENGTH_OP_H_
#define DATASET_ENGINE_DATASETOPS_BUCKET_BATCH_BY_LENGTH_OP_H_

#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "dataset/core/config_manager.h"
#include "dataset/core/tensor.h"
#include "dataset/engine/dataset_iterator.h"
#include "dataset/engine/datasetops/batch_op.h"
#include "dataset/engine/datasetops/pipeline_op.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
class DataBuffer;

class BucketBatchByLengthOp : public PipelineOp {
 public:
  class Builder {
   public:
    Builder(std::vector<std::string> length_dependent_columns, std::vector<int32_t> bucket_boundaries,
            std::vector<int32_t> bucket_batch_sizes);

    ~Builder() = default;

    Builder &SetLengthDependentColumns(std::vector<std::string> length_dependent_columns) {
      builder_length_dependent_columns_ = length_dependent_columns;
      return *this;
    }

    Builder &SetBucketBoundaries(std::vector<int32_t> bucket_boundaries) {
      builder_bucket_boundaries_ = bucket_boundaries;
      return *this;
    }

    Builder &SetBucketBatchSizes(std::vector<int32_t> bucket_batch_sizes) {
      builder_bucket_batch_sizes_ = bucket_batch_sizes;
      return *this;
    }

    Builder &SetElementLengthFunction(py::function element_length_function) {
      builder_element_length_function_ = element_length_function;
      return *this;
    }

    Builder &SetPadInfo(PadInfo pad_info) {
      builder_pad_info_ = pad_info;
      return *this;
    }

    Builder &SetPadToBucketBoundary(bool pad_to_bucket_boundary) {
      builder_pad_to_bucket_boundary_ = pad_to_bucket_boundary;
      return *this;
    }

    Builder &SetDropRemainder(bool drop_remainder) {
      builder_drop_remainder_ = drop_remainder;
      return *this;
    }

    Builder &SetOpConnectorSize(int32_t op_connector_size) {
      builder_op_connector_size_ = op_connector_size;
      return *this;
    }

    Status Build(std::shared_ptr<BucketBatchByLengthOp> *new_bucket_batch_by_length_op);

   private:
    Status SanityCheck();

    std::vector<std::string> builder_length_dependent_columns_;
    std::vector<int32_t> builder_bucket_boundaries_;
    std::vector<int32_t> builder_bucket_batch_sizes_;
    py::function builder_element_length_function_;
    PadInfo builder_pad_info_;
    bool builder_pad_to_bucket_boundary_;
    bool builder_drop_remainder_;
    int32_t builder_op_connector_size_;
  };

  BucketBatchByLengthOp(std::vector<std::string> length_dependent_columns, std::vector<int32_t> bucket_boundaries,
                        std::vector<int32_t> bucket_batch_sizes, py::function element_length_function, PadInfo pad_info,
                        bool pad_to_bucket_boundary, bool drop_remainder, int32_t op_connector_size);

  // Might need to batch remaining buckets after receiving eoe, so override this method.
  // @param int32_t workerId
  // @return Status - The error code returned
  Status EoeReceived(int32_t) override;

  // A print method typically used for debugging
  // @param out - The output stream to write output to
  // @param show_all - A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override;

  // << Stream output operator overload
  // @notes This allows you to write the debug print info using stream operators
  // @param out - reference to the output stream being overloaded
  // @param sO - reference to the BucketBatchByLengthOp to display
  // @return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const BucketBatchByLengthOp &bo) {
    bo.Print(out, false);
    return out;
  }

  // Main loop of batch
  // @return Status - The error code returned
  Status operator()() override;

  // Function that is called by ResetOp at the end of every epoch
  // @return Status - The error code returned
  Status Reset() override;

 private:
  Status ObtainElementLength(int32_t *out_element_length, TensorRow element);

  Status PadAndBatchBucket(int32_t bucket_index, int32_t batch_size);

  std::vector<std::string> length_dependent_columns_;
  std::vector<int32_t> bucket_boundaries_;
  std::vector<int32_t> bucket_batch_sizes_;
  py::function element_length_function_;
  PadInfo pad_info_;
  bool pad_to_bucket_boundary_;
  bool drop_remainder_;

  int32_t batch_count_;
  std::unique_ptr<ChildIterator> child_iterator_;
  std::vector<std::unique_ptr<TensorQTable>> buckets_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_DATASETOPS_BUCKET_BATCH_BY_LENGTH_OP_H_
