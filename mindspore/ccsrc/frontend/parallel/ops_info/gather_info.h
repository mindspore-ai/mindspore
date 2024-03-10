/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_GATHER_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_GATHER_INFO_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"
#include "ir/value.h"
#include "utils/hash_map.h"

namespace mindspore {
namespace parallel {
constexpr char BATCH_DIMS[] = "batch_dims";
enum GatherMode {
  BATCH = 0,
  NORMAL,
  MANUAL,
  SHARD_BATCH_AND_AXIS,
  SHARD_AXIS_0_DYNAMIC,
  SHARD_AXIS_0_STATIC,
  SHARD_AXIS_1,
  INVALID
};

class GatherUtil;
using GatherUtilPtr = std::shared_ptr<GatherUtil>;

class GatherUtil {
 public:
  GatherUtil(std::string name, Shapes inputs_shape, Shapes outputs_shape, int64_t axis)
      : name_(std::move(name)),
        inputs_shape_(std::move(inputs_shape)),
        outputs_shape_(std::move(outputs_shape)),
        axis_(axis) {}
  virtual ~GatherUtil() = default;
  virtual Status CheckStrategy(const Shape &param_strategy, const Shape &indices_strategy) = 0;
  virtual Status InferForwardCommunication() { return SUCCESS; }
  virtual Status InferTensorInfo() = 0;
  virtual Status InferDevMatrixShape() = 0;
  virtual Status InferTensorMap() = 0;
  virtual Status InferReplaceGraph(const CNodePtr &cnode) { return SUCCESS; }
  virtual Status InferReplaceOps() { return SUCCESS; }

  void set_param_strategy(const Shape &a) { param_strategy_ = a; }
  void set_indices_strategy(const Shape &a) { indices_strategy_ = a; }
  void set_gather_mode(const GatherMode &a) { gather_mode_ = a; }
  GatherMode gather_mode() const { return gather_mode_; }
  Shape dev_matrix_shape() const { return dev_matrix_shape_; }
  void set_dev_matrix_shape(const Shape &a) { dev_matrix_shape_ = a; }
  TensorMaps inputs_tensor_map() const { return inputs_tensor_map_; }
  TensorMaps outputs_tensor_map() const { return outputs_tensor_map_; }
  void set_inputs_tensor_map(const TensorMaps &a) { inputs_tensor_map_ = a; }
  void set_outputs_tensor_map(const TensorMaps &a) { outputs_tensor_map_ = a; }
  std::vector<TensorInfo> inputs_tensor_info() const { return inputs_tensor_info_; }
  std::vector<TensorInfo> outputs_tensor_info() const { return outputs_tensor_info_; }
  ForwardOp forward_op() const { return forward_op_; }
  ForwardOp replace_op() const { return replace_op_; }
  ReplaceGraphPtr replace_graph() const { return replace_graph_; }
  bool repeated_num_in_dev_matrix_right() const { return repeated_num_in_dev_matrix_right_; }
  Shape out_dev_matrix_shape() const { return out_dev_matrix_shape_; }
  std::string GatherModeToString() const { return gather_mode_string_[gather_mode_]; }

 protected:
  std::string name_;
  Shapes inputs_shape_;
  Shapes outputs_shape_;
  int64_t axis_;

  Shape param_strategy_;
  Shape indices_strategy_;
  GatherMode gather_mode_ = INVALID;
  Shape dev_matrix_shape_;
  TensorMaps inputs_tensor_map_;
  TensorMaps outputs_tensor_map_;
  std::vector<TensorInfo> inputs_tensor_info_;
  std::vector<TensorInfo> outputs_tensor_info_;
  ForwardOp forward_op_;
  ForwardOp replace_op_;
  ReplaceGraphPtr replace_graph_;

  Status InferTensorInfoNoSplitAxis();
  bool repeated_num_in_dev_matrix_right_ = true;  // only for shard axis
  Shape out_dev_matrix_shape_;                    // only for shard axis

 private:
  const std::vector<std::string> gather_mode_string_ = {
    "batch",        "normal", "manual", "shard_batch_and_axis", "shard_axis_0_dynamic", "shard_axis_0_static",
    "shard_axis_1", "invalid"};
};

// batch mode: batch_dims > 1
// constraint:
//   1) axis can not be split
//   2) can not set out_strategy
// param  shape: [A, B, C, D，E]
// indices shape: [A, B, F, G]
// batch_dims = 2
// axis = 3
// out = gather(param,  indices,  axis)
// out shape: [A, B, C, F, G, E]
// parameter's strategy: [a, b, c, 1, e], indices' strategy: [a, b, f, g]
// output's strategy: [a, b, c, f, g, e]
// dev_matrix: [a, b, f, g, c, 1, e]
class BatchImpl : public GatherUtil {
 public:
  BatchImpl(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, int64_t axis)
      : GatherUtil(name, inputs_shape, outputs_shape, axis) {}
  ~BatchImpl() override = default;
  Status CheckStrategy(const Shape &param_strategy, const Shape &indices_strategy) override;
  Status InferDevMatrixShape() override;
  void set_batch_dims(int64_t batch_dims) { batch_dims_ = batch_dims; }
  Status InferTensorMap() override;
  Status InferTensorInfo() override { return InferTensorInfoNoSplitAxis(); }

 private:
  int64_t batch_dims_ = 0;
};

// normal mode: batch_dims = 0, and the axis has not be split
// constraint:
//   1) can not set out_strategy
// param  shape: [C, D，E]
// indices shape: [F, G]
// batch_dims = 0
// axis = 1
// out = gather(param,  indices,  axis)
// out shape: [C, F, G, E]
// parameter's strategy: [c, 1, e], indices' strategy: [f, g]
// output's strategy: [c, f, g, e]
// dev_matrix: [f, g, c, 1, e]
class NormalImpl : public GatherUtil {
 public:
  NormalImpl(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, int64_t axis)
      : GatherUtil(name, inputs_shape, outputs_shape, axis) {}
  ~NormalImpl() override = default;
  Status CheckStrategy(const Shape &param_strategy, const Shape &indices_strategy) override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status InferTensorInfo() override { return InferTensorInfoNoSplitAxis(); }
};

// manual mode: the primitive has the "manual_split" attr, axis = 0, batch_dims = 0
// constraint:
//   1) the field dimension of indices is the last dimension;
//   2) can not support repeated calculation
//   3) parameter's dim >= 1, indices' dim >= 1
//   4) can not set out_strategy
// param  shape: [A, B, ..., C]
// indices shape: [D, ..., E, F]
// batch_dims = 0
// axis = 0
// out = gather(param,  indices,  axis)
// out shape: [D, ..., E, F, B, ..., C]
// parameter's strategy: [a, b, ..., c], indices' strategy: [1, ..., 1, a]
// output's strategy: [1, ..., 1, a, b, ..., c]
// dev_matrix: [a, b, ..., c]
class ManualImpl : public GatherUtil {
 public:
  ManualImpl(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, int64_t axis)
      : GatherUtil(name, inputs_shape, outputs_shape, axis) {}
  ~ManualImpl() override = default;
  Status CheckStrategy(const Shape &param_strategy, const Shape &indices_strategy) override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status InferTensorInfo() override;
  Status InferReplaceGraph(const CNodePtr &cnode) override;
  Status InferReplaceOps() override;

  void set_param_split_shapes(const Shape &a) { param_split_shapes_ = a; }
  void set_index_offsets(const Shape &a) { index_offsets_ = a; }
  void set_target(const std::string &a) { target_ = a; }
  void set_attrs(const mindspore::HashMap<std::string, ValuePtr> &a) { attrs_ = a; }
  void set_replace_op_name(const std::string &a) { replace_op_name_ = a; }

 protected:
  Status InferOffset();
  std::string target_ = DEVICE;
  mindspore::HashMap<std::string, ValuePtr> attrs_;
  std::string replace_op_name_;
  int64_t index_offset_ = 0;

 private:
  Shape param_split_shapes_;
  Shape index_offsets_;
};

class GatherManualImpl : public ManualImpl {
 public:
  GatherManualImpl(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, int64_t axis)
      : ManualImpl(name, inputs_shape, outputs_shape, axis) {}
  ~GatherManualImpl() override = default;
  Status InferReplaceGraph(const CNodePtr &cnode) override;
};

// SHARD_AXIS_0_DYNAMIC, SHARD_AXIS_0_STATIC and SHARD_AXIS_1 mode: batch_dims = 0, and split axis
// constraint:
//   1) parameter's dim is 1 or 2, indices' dim >= 1
//   2) indices can't be split
//   3) axis = 0 or axis = 1
//   4) if axis = 1, can not support repeated calculation
//   5) if axis = 0, and param_shape[1] is split, can not support repeated calculation
// param  shape: [A, B]
// indices shape: [C, D]
// batch_dims = 0
// axis = 0
// out = gather(param,  indices,  axis)
// out shape: [A, B, C]
// parameter's strategy: [a, b], indices' strategy: [1, 1]
// output's strategy:
//   1) if use allreduce: [1, 1, b]
//   2) if use reducescatter: [a, 1, b]
// dev_matrix: [a, b]
class ShardAxisImpl : public GatherUtil {
 public:
  ShardAxisImpl(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, int64_t axis)
      : GatherUtil(name, inputs_shape, outputs_shape, axis) {}
  ~ShardAxisImpl() override = default;
  Status CheckStrategy(const Shape &param_strategy, const Shape &indices_strategy) override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status InferTensorInfo() override;
  virtual Status InferBias();
  void set_target(const std::string &a) { target_ = a; }
  void set_dynamic_shape_indices(bool a) { dynamic_shape_indices_ = a; }
  void set_attrs(const mindspore::HashMap<std::string, ValuePtr> &a) { attrs_ = a; }
  void set_replace_op_name(const std::string &a) { replace_op_name_ = a; }
  void set_axis_split_forward_allreduce(bool a) { axis_split_forward_allreduce_ = a; }

  // ShardBatchAndAxisImpl and ShardAxisImpl
  Status InferForwardCommunication() override;
  Status InferReplaceOps() override;
  Status InferReplaceGraph(const CNodePtr &cnode) override;
  void set_assigned_parallel(bool is_assigned_parallel) { is_assigned_parallel_ = is_assigned_parallel; }

 protected:
  // use for split axis
  Status CheckSplitAxisStrategy(const Shape &param_strategy, const Shape &indices_strategy);
  void SetAttribute(const Shape &param_strategy);
  Status InferGroup();
  std::string target_ = DEVICE;
  std::string replace_op_name_;
  bool dynamic_shape_indices_ = false;
  bool is_assigned_parallel_ = false;
  bool axis_split_forward_allreduce_ = false;  // when axis is split, use reducescatter as default in forward
  int64_t repeated_calculation_num_ = 1;
  Group group_;
  mindspore::HashMap<std::string, ValuePtr> attrs_;
  int64_t bias_ = 0;
  int64_t slice_size_ = 0;
};

// shard_batch_and_axis mode: axis = 0, batch_dims = 0, and only split the first dimension of parameter and the first
// dimension of indices constraint:
//   1) the dim of param is 2, and the dim of indices is 2;
//   2) only split the first dimension of parameter and the first dimension of indices, other dims can not be split
//   3) do not support repeat calculation
// param  shape: [A, B]
// indices shape: [C, D]
// batch_dims = 0
// axis = 0
// out = gather(param,  indices,  axis)
// out shape: [C, D, B]
// parameter's strategy: [a, 1], indices' strategy: [c, 1]
// output's strategy:
//   1) if use allreduce: [c, 1, 1]
//   2) if use reducescatter: [a*c, 1, 1]
// dev_matrix:
//   1) if use allreduce: [c, a]
//   2) if use reducescatter: [a*c]
class ShardBatchAndAxisImpl : public ShardAxisImpl {
 public:
  ShardBatchAndAxisImpl(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, int64_t axis)
      : ShardAxisImpl(name, inputs_shape, outputs_shape, axis) {}
  ~ShardBatchAndAxisImpl() override = default;
  Status CheckStrategy(const Shape &param_strategy, const Shape &indices_strategy) override {
    return SUCCESS;
  }  // no need check
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status InferTensorInfo() override { return InferTensorInfoNoSplitAxis(); }  // do not need to use out_dev_matrix_shape
  Status InferBias() override;
};

class GatherInfo : public OperatorInfo {
 public:
  GatherInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs, const std::string &replace_op_name = GATHERV2)
      : OperatorInfo(name, inputs_shape, outputs_shape, attrs, std::make_shared<GatherCost>()),
        replace_op_name_(replace_op_name) {}
  ~GatherInfo() override = default;
  Status Init(const StrategyPtr &in_strategy, const StrategyPtr &out_strategy,
              const std::vector<std::shared_ptr<TensorLayout>> &in_tensor_layouts = {},
              const std::vector<std::shared_ptr<TensorLayout>> &out_tensor_layouts = {}) override;
  Status InitForCostModel(const StrategyPtr &in_strategy, const StrategyPtr &out_strategy) override;

  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;
  ReplaceGraphPtr replace_graph(const CNodePtr &cnode) override;
  std::shared_ptr<Strategies> GenerateBatchStrategies() override;
  const std::vector<int64_t> &param_split_shapes() const { return param_split_shapes_; }
  const std::vector<int64_t> &index_offsets() const { return index_offsets_; }

 protected:
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status CheckOutputStrategy(const StrategyPtr &out_strategy) override;
  Status InferMirrorOps() override;
  Status InferForwardCommunication() override;
  Status InferTensorInfo() override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status GetAttrs() override;
  virtual void DealWithBatchDimsMirrorOp() noexcept;
  virtual void GetBatchDims() noexcept;
  virtual GatherUtilPtr MakeManualUtil();
  int64_t axis_ = 0;

 private:
  GatherMode GetGatherMode(const Shape &param_strategy, const Shape &indices_strategy) const;
  int64_t batch_dims_ = 0;
  Status GetManualSplitAttr();
  Status GetManualSplitWithoutOffsetAttr();
  Status ComputeReplaceOp();
  bool ShardBatchAndAxis(const Shape &param_strategy, const Shape &indices_strategy) const;

  std::string target_ = DEVICE;
  int64_t bias_ = 0;
  std::string replace_op_name_ = GATHERV2;
  bool manual_split_ = false;
  bool dynamic_shape_indices_ = false;
  std::vector<int64_t> param_split_shapes_;  // manual split
  std::vector<int64_t> index_offsets_;       // manual split
  GatherMode gather_mode_ = INVALID;
  GatherUtilPtr gather_util_;
};

class SparseGatherV2Info final : public GatherInfo {
 public:
  SparseGatherV2Info(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                     const PrimitiveAttrs &attrs, const std::string &replace_op_name = SPARSE_GATHERV2)
      : GatherInfo(name, inputs_shape, outputs_shape, attrs, replace_op_name) {}
  ~SparseGatherV2Info() override = default;

 protected:
  void DealWithBatchDimsMirrorOp() noexcept override {}
  void GetBatchDims() noexcept override {}
  GatherUtilPtr MakeManualUtil() override;
};

class EmbeddingLookupInfo final : public GatherInfo {
 public:
  EmbeddingLookupInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                      const PrimitiveAttrs &attrs)
      : GatherInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~EmbeddingLookupInfo() override = default;

 protected:
  void DealWithBatchDimsMirrorOp() noexcept override {}
  void GetBatchDims() noexcept override {}
  GatherUtilPtr MakeManualUtil() override;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_GATHER_INFO_H_
