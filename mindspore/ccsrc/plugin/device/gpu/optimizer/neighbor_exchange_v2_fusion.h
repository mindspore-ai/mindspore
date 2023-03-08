/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_NEIGHBOR_EXCHANGE_V2_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_NEIGHBOR_EXCHANGE_V2_FUSION_H_

#include <memory>
#include <vector>
#include "backend/common/optimizer/optimizer.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
struct SplitvNodeInfo {
  bool is_first = false;
  bool is_last = false;
  int64_t split_dim = 0;
  int64_t num_split = 0;
  TypeId dtype = kTypeUnknown;
  ShapeVector base_shape;
  std::vector<int64_t> send_lens = {};
  std::vector<ShapeVector> shapes = {};
  std::vector<int64_t> size_splits = {};
};

struct SliceNodeInfo {
  ShapeVector base_shape;
  int64_t slice_dim = 0;
  TypeId input_dtype = kTypeUnknown;
  int64_t slice_begin = 0;
  int64_t slice_size = 0;
};

class NeighborExchangeV2Fusion : public PatternProcessPass {
 public:
  explicit NeighborExchangeV2Fusion(bool multigraph = true)
      : PatternProcessPass("neighbor_exchange_v2_fusion", multigraph) {}
  ~NeighborExchangeV2Fusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  std::vector<CNodePtr> CreateSplitNodes(const FuncGraphPtr &graph, const CNodePtr &neighbor_exchange_v2,
                                         std::vector<int64_t> *split_num) const;
  CNodePtr CreateConcatNode(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &concat_input,
                            const std::vector<ShapeVector> &output_shape, const std::vector<TypeId> &output_dtype,
                            int64_t axis, int64_t input_nums) const;
  CNodePtr CreateLeftRightConcat(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &all_to_all_v_outputs,
                                 const std::vector<int64_t> &recv_rank_ids, const std::vector<int64_t> &recv_lens,
                                 bool is_left) const;
  CNodePtr CreateMiddleConcat(const FuncGraphPtr &graph, const CNodePtr &neighbor_exchange_v2,
                              const std::vector<AnfNodePtr> &all_to_all_v_outputs,
                              const std::vector<int64_t> &recv_rank_ids, const std::vector<int64_t> &recv_lens,
                              int64_t concat_dim) const;
  CNodePtr AllToAllvRecvEmpty(const FuncGraphPtr &graph, const CNodePtr &neighbor_exchange_v2,
                              const CNodePtr &all_to_all_v) const;
  CNodePtr CreateConcatNodes(const FuncGraphPtr &graph, const CNodePtr &neighbor_exchange_v2,
                             const CNodePtr &all_to_all_v) const;
};

class NeighborExchangeV2GradFusion : public PatternProcessPass {
 public:
  explicit NeighborExchangeV2GradFusion(bool multigraph = true)
      : PatternProcessPass("neighbor_exchange_v2_grad_fusion", multigraph) {}
  ~NeighborExchangeV2GradFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  std::vector<CNodePtr> CreateSplitNodesForGrad(const FuncGraphPtr &graph, const CNodePtr &neighbor_exchange_v2_grad,
                                                std::vector<int64_t> *split_num) const;
  CNodePtr CreatePadNode(const FuncGraphPtr &graph, const AnfNodePtr &input, const std::vector<int64_t> &begin,
                         const std::vector<int64_t> &size, const ShapeVector &shape, TypeId dtype) const;
  CNodePtr CreateSplitGradNodes(const FuncGraphPtr &graph, const CNodePtr &neighbor_exchange_v2_grad,
                                const CNodePtr &all_to_all_v, const std::vector<CNodePtr> &split_nodes,
                                const std::vector<int64_t> &split_num) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_NEIGHBOR_EXCHANGE_V2_FUSION_H_
