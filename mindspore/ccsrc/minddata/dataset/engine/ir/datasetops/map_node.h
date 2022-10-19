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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_MAP_NODE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_MAP_NODE_H_

#include <memory>
#include <string>
#include <vector>

#ifdef ENABLE_PYTHON
#include "minddata/dataset/api/python/python_mp.h"
#endif
#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"

namespace mindspore {
namespace dataset {
class MapNode : public DatasetNode {
 public:
  /// \brief Constructor
  MapNode(std::shared_ptr<DatasetNode> child, std::vector<std::shared_ptr<TensorOperation>> operations,
          std::vector<std::string> input_columns = {}, std::vector<std::string> output_columns = {},
          std::shared_ptr<DatasetCache> cache = nullptr, std::vector<std::shared_ptr<DSCallback>> callbacks = {},
          ManualOffloadMode offload = ManualOffloadMode::kUnspecified,
          std::shared_ptr<PythonMultiprocessingRuntime> python_mp = nullptr);

  /// \brief Destructor
  ~MapNode() override = default;

  /// \brief Node name getter
  /// \return Name of the current node
  std::string Name() const override { return kMapNode; }

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

  /// \brief clear all callbacks
  void ClearCallbacks() { callbacks_.clear(); }

  /// \brief getter to get all tensor operations
  std::vector<std::shared_ptr<TensorOperation>> operations();

  /// \brief setter to set all tensor operations
  void setOperations(const std::vector<std::shared_ptr<TensorOperation>> &operations);

  /// \brief Getter functions
  /// \brief Getter of tensor operations
  /// \return Vector of operations the Map node will process
  const auto &TensorOperations() const { return operations_; }
  const std::vector<std::string> &InputColumns() const { return input_columns_; }
  const std::vector<std::string> &OutputColumns() const { return output_columns_; }
  const std::vector<std::shared_ptr<DSCallback>> &Callbacks() const { return callbacks_; }
  ManualOffloadMode GetOffload() const { return offload_; }

  /// \brief setter to set offload flag of node
  void SetOffload(ManualOffloadMode offload);

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *out_json) override;

#ifndef ENABLE_ANDROID
  /// \brief Function for read dataset operation from json
  /// \param[in] json_obj The JSON object to be deserialized
  /// \param[in] ds dataset node constructed
  /// \param[out] result Deserialized dataset after the operation
  /// \return Status The status code returned
  static Status from_json(nlohmann::json json_obj, std::shared_ptr<DatasetNode> ds,
                          std::shared_ptr<DatasetNode> *result);
#endif

  /// \brief Base-class override for GetDatasetSize
  /// \param[in] size_getter Shared pointer to DatasetSizeGetter
  /// \param[in] estimate This is only supported by some of the ops and it's used to speed up the process of getting
  ///     dataset size at the expense of accuracy.
  /// \param[out] dataset_size the size of the dataset
  /// \return Status of the function
  Status GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                        int64_t *dataset_size) override;

 private:
  std::vector<std::shared_ptr<TensorOperation>> operations_;
  std::vector<std::string> input_columns_;
  std::vector<std::string> output_columns_;
  std::vector<std::shared_ptr<DSCallback>> callbacks_;

  /// \brief ManualOffloadMode to indicate manual_offload status
  ManualOffloadMode offload_;

  std::shared_ptr<PythonMultiprocessingRuntime> python_mp_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_MAP_NODE_H_
