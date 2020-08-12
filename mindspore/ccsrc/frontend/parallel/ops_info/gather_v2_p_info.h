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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_GATHER_V2_P_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_GATHER_V2_P_INFO_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ir/value.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
class GatherV2PInfo : public OperatorInfo {
 public:
  GatherV2PInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : OperatorInfo(name, inputs_shape, outputs_shape, attrs, std::make_shared<GatherV2PCost>()),
        axis_(0),
        bias_(0),
        index_offset_(0),
        slice_size_(0) {}
  ~GatherV2PInfo() override = default;
  Status Init(const StrategyPtr &strategy) override;
  Status InitForCostModel(const StrategyPtr &strategy) override;

  Status GenerateStrategies(int32_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;
  ReplaceGraphPtr replace_graph(const CNodePtr &cnode) override;
  std::shared_ptr<Strategys> GenerateBatchStrategies() override;
  const std::vector<int64_t> &param_split_shapes() const { return param_split_shapes_; }
  const std::vector<int64_t> &index_offsets() const { return index_offsets_; }

 protected:
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferMirrorOps() override;
  Status InferForwardCommunication() override;
  Status InferTensorInfo() override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status GetAttrs() override;

  Status ComputeReplaceGraph(const CNodePtr &cnode);
  Status CheckManualSplit(const Strategys &strategy);
  Status GetManualSplitAttr();
  Status GetManualSplitWithoutOffsetAttr();
  Status ComputeReplaceOp();
  Status InferBias();
  Status InferOffset();
  Status InferGroup();

  int32_t axis_;
  std::string target_ = DEVICE;
  std::string replace_op_name_ = GATHERV2;
  int64_t bias_;
  int64_t index_offset_;
  int64_t slice_size_;
  Shape out_dev_matrix_shape_;
  Group group_;
  bool manual_split_ = false;
  std::vector<int64_t> param_split_shapes_;
  std::vector<int64_t> index_offsets_;
};

class SparseGatherV2Info : public GatherV2PInfo {
 public:
  SparseGatherV2Info(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                     const PrimitiveAttrs &attrs)
      : GatherV2PInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~SparseGatherV2Info() override = default;

 private:
  std::string replace_op_name_ = SPARSE_GATHERV2;
};

class EmbeddingLookupInfo : public GatherV2PInfo {
 public:
  EmbeddingLookupInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                      const PrimitiveAttrs &attrs)
      : GatherV2PInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~EmbeddingLookupInfo() override = default;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_GATHER_V2_P_INFO_H_
