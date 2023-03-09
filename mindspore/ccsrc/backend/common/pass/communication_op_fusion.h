/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_COMMUNICATION_OP_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_COMMUNICATION_OP_FUSION_H_
#include <utility>
#include <vector>
#include <string>
#include "include/backend/visible.h"
#include "include/backend/optimizer/pass.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace opt {
struct CommunicationOpInfo {
  std::vector<CNodePtr> communication_op_nodes;
  std::vector<float> input_grad_size;
  std::vector<float> input_grad_time;
};

class BACKEND_EXPORT CommunicationOpFusion : public Pass {
 public:
  explicit CommunicationOpFusion(const std::string &name, std::string op_name, size_t groups = 1)
      : Pass(name), op_name_(std::move(op_name)), groups_(groups) {}
  ~CommunicationOpFusion() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  bool DoFusion(const FuncGraphPtr &func_graph, const CommunicationOpInfo &communication_op_info,
                const std::vector<size_t> &segment_index) const;
  void GetAllReduceSplitSegment(const std::vector<CNodePtr> &nodes, int64_t threshold,
                                std::vector<size_t> *segment_index) const;
  AnfNodePtr CreateFusedCommunicationOp(const FuncGraphPtr &func_graph,
                                        const CommunicationOpInfo &communication_op_info, size_t start_index,
                                        size_t end_index) const;
  bool GetSplitSegments(const CommunicationOpInfo &communication_op_info, std::vector<size_t> *segment_index,
                        const std::string &group) const;
  std::string op_name_;
  size_t groups_ = 1;
};

class SendFusion : public CommunicationOpFusion {
 public:
  explicit SendFusion(size_t groups = 1) : CommunicationOpFusion("send_fusion", kHcomSendOpName, groups) {}
  ~SendFusion() override = default;
};

class RecvFusion : public CommunicationOpFusion {
 public:
  explicit RecvFusion(size_t groups = 1) : CommunicationOpFusion("recv_fusion", kReceiveOpName, groups) {}
  ~RecvFusion() override = default;
};

class AllReduceFusion : public CommunicationOpFusion {
 public:
  explicit AllReduceFusion(size_t groups = 1) : CommunicationOpFusion("all_reduce_fusion", kAllReduceOpName, groups) {}
  ~AllReduceFusion() override = default;
};

class AllGatherFusion : public CommunicationOpFusion {
 public:
  explicit AllGatherFusion(size_t groups = 1) : CommunicationOpFusion("all_gather_fusion", kAllGatherOpName, groups) {}
  ~AllGatherFusion() override = default;
};

class BroadcastFusion : public CommunicationOpFusion {
 public:
  explicit BroadcastFusion(size_t groups = 1) : CommunicationOpFusion("broadcast_fusion", kBroadcastOpName, groups) {}
  ~BroadcastFusion() override = default;
};

class ReduceScatterFusion : public CommunicationOpFusion {
 public:
  explicit ReduceScatterFusion(size_t groups = 1)
      : CommunicationOpFusion("reduce_scatter_fusion", kReduceScatterOpName, groups) {}
  ~ReduceScatterFusion() override = default;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_COMMUNICATION_OP_FUSION_H_
