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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_BATCH_NODE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_BATCH_NODE_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"
#include "minddata/dataset/engine/opt/pass.h"

namespace mindspore {
namespace dataset {

class BatchNode : public DatasetNode {
 public:
#ifdef ENABLE_PYTHON
  /// \brief Constructor #1, for Python API to create a BatchNode
  BatchNode(std::shared_ptr<DatasetNode> child, int32_t batch_size, bool drop_remainder, bool pad,
            const std::vector<std::string> &in_col_names, const std::vector<std::string> &out_col_names,
            const std::vector<std::string> &col_order, py::function batch_size_func, py::function batch_map_func,
            std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>> pad_map);
#endif

  /// \brief Constructor #2 for C++ API to create a BatchNode
  BatchNode(std::shared_ptr<DatasetNode> child, int32_t batch_size, bool drop_remainder);

  /// \brief Destructor
  ~BatchNode() = default;

  /// \brief Node name getter
  /// \return Name of the current node
  std::string Name() const override { return kBatchNode; }

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

  /// \brief Base-class override for GetDatasetSize
  /// \param[in] size_getter Shared pointer to DatasetSizeGetter
  /// \param[in] estimate This is only supported by some of the ops and it's used to speed up the process of getting
  ///     dataset size at the expense of accuracy.
  /// \param[out] dataset_size the size of the dataset
  /// \return Status of the function
  Status GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                        int64_t *dataset_size) override;

  /// \brief Base-class override for accepting IRNodePass visitor
  /// \param[in] p The node to visit
  /// \param[out] modified Indicator if the node was modified
  /// \return Status of the node visit
  Status Accept(IRNodePass *const p, bool *const modified) override;

  /// \brief Base-class override for accepting IRNodePass visitor
  /// \param[in] p The node to visit
  /// \param[out] modified Indicator if the node was modified
  /// \return Status of the node visit
  Status AcceptAfter(IRNodePass *const p, bool *const modified) override;

  /// \brief Getter functions
  int32_t BatchSize() const { return batch_size_; }
  bool DropRemainder() const { return drop_remainder_; }
#ifdef ENABLE_PYTHON
  bool Pad() const { return pad_; }
  const std::vector<std::string> &InColNames() const { return in_col_names_; }
  const std::vector<std::string> &OutColNames() const { return out_col_names_; }
  const std::vector<std::string> &ColOrder() const { return col_order_; }
  const py::function &BatchSizeFunc() const { return batch_size_func_; }
  const py::function &BatchMapFunc() const { return batch_map_func_; }
  const std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>> &PadMap() const { return pad_map_; }
#endif

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t batch_size_;
  bool drop_remainder_;
  bool pad_;
  std::vector<std::string> in_col_names_;
  std::vector<std::string> out_col_names_;
  std::vector<std::string> col_order_;
#ifdef ENABLE_PYTHON
  py::function batch_size_func_;
  py::function batch_map_func_;
#endif
  std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>> pad_map_;
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_BATCH_NODE_H_
