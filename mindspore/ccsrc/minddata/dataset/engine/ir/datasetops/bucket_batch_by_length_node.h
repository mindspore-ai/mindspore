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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_BUCKET_BATCH_BY_LENGTH_NODE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_BUCKET_BATCH_BY_LENGTH_NODE_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"

namespace mindspore {
namespace dataset {

class BucketBatchByLengthNode : public DatasetNode {
 public:
  /// \brief Constructor
  BucketBatchByLengthNode(std::shared_ptr<DatasetNode> child, const std::vector<std::string> &column_names,
                          const std::vector<int32_t> &bucket_boundaries, const std::vector<int32_t> &bucket_batch_sizes,
                          std::shared_ptr<TensorOp> element_length_function = nullptr,
                          const std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>> &pad_info = {},
                          bool pad_to_bucket_boundary = false, bool drop_remainder = false);

  /// \brief Destructor
  ~BucketBatchByLengthNode() = default;

  /// \brief Node name getter
  /// \return Name of the current node
  std::string Name() const override { return kBucketBatchByLengthNode; }

  /// \brief Print the description
  /// \param out - The output stream to write output to
  void Print(std::ostream &out) const override;

  /// \brief Copy the node to a new object
  /// \return A shared pointer to the new copy
  std::shared_ptr<DatasetNode> Copy() override;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \param node_ops - A vector containing shared pointer to the Dataset Ops that this object will create
  /// \return Status Status::OK() if build successfully
  Status Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) override;

  /// \brief Parameters validation
  /// \return Status Status::OK() if all the parameters are valid
  Status ValidateParams() override;

  bool IsSizeDefined() override { return false; };

  /// \brief Getter functions
  const std::vector<std::string> &ColumnNames() const { return column_names_; }
  const std::vector<int32_t> &BucketBoundaries() const { return bucket_boundaries_; }
  const std::vector<int32_t> &BucketBatchSizes() const { return bucket_batch_sizes_; }
  const std::shared_ptr<TensorOp> &ElementLengthFunction() const { return element_length_function_; }
  const std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>> &PadInfo() const { return pad_info_; }
  bool PadToBucketBoundary() const { return pad_to_bucket_boundary_; }
  bool DropRemainder() const { return drop_remainder_; }

 private:
  std::vector<std::string> column_names_;
  std::vector<int32_t> bucket_boundaries_;
  std::vector<int32_t> bucket_batch_sizes_;
  std::shared_ptr<TensorOp> element_length_function_;
  std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>> pad_info_;
  bool pad_to_bucket_boundary_;
  bool drop_remainder_;
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_BUCKET_BATCH_BY_LENGTH_NODE_H_
