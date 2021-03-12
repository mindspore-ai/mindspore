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
#include "backend/optimizer/ascend/ir_fusion/batchnorm_to_batchnorm3d.h"
#include <memory>
#include <string>
#include <vector>
#include "backend/session/anf_runtime_algorithm.h"
#include "ir/primitive.h"
#include "utils/utils.h"
#include "base/core_ops.h"
#include "abstract/abstract_value.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kBN3InputXIndex = 1;
constexpr size_t kBn3DTrainInputTensorNum = 3;
CNodePtr CreateBatchNorm3D(const FuncGraphPtr &graph, const CNodePtr &batchnorm) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(batchnorm);
  auto prim = std::make_shared<Primitive>(kBatchNorm3DOpName);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim)};
  auto is_training = AnfAlgo::GetNodeAttr<bool>(batchnorm, kAttrIsTraining);
  for (size_t i = 1; i < batchnorm->size(); ++i) {
    if (is_training && i > kBn3DTrainInputTensorNum) {
      continue;
    } else {
      inputs.push_back(batchnorm->input(i));
    }
  }
  auto new_node = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_scope(batchnorm->scope());
  new_node->set_abstract(batchnorm->abstract());
  AnfAlgo::CopyNodeAttrs(batchnorm, new_node);
  return new_node;
}

bool NeedFusion(const FuncGraphPtr &graph, const CNodePtr &batchnorm) {
  MS_EXCEPTION_IF_NULL(batchnorm);
  if (!AnfAlgo::HasNodeAttr(kAttrIsTraining, batchnorm)) {
    MS_LOG(INFO) << "BatchNorm has no is_training attr.";
    return false;
  }
  auto is_training = AnfAlgo::GetNodeAttr<bool>(batchnorm, kAttrIsTraining);
  auto format = AnfAlgo::GetNodeAttr<std::string>(batchnorm, kAttrFormat);
  if (is_training && format == kOpFormat_NCDHW) {
    if (AnfAlgo::GetInputTensorNum(batchnorm) < kBn3DTrainInputTensorNum) {
      MS_LOG(INFO) << "When data format is NCDHW and is_training is true, BatchNorm's input less than "
                   << kBn3DTrainInputTensorNum;
      return false;
    }
  } else {
    if (AnfAlgo::GetInputTensorNum(batchnorm) < kBnInputTensorNum) {
      MS_LOG(INFO) << "BatchNorm's input less than " << kBnInputTensorNum;
      return false;
    }
  }
  const auto &ori_inputs = batchnorm->inputs();
  auto x_shape = AnfAlgo::GetOutputInferShape(ori_inputs[kBN3InputXIndex], 0);
  if (format != kOpFormat_NCDHW || x_shape.size() != 5) {
    MS_LOG(INFO) << "Only format is NCDHW and the input dim of BatchNorm is 5, then do fusion. But format is: "
                 << format << ", size of x_shape is: " << x_shape.size();
    return false;
  }
  return true;
}
}  // namespace

const BaseRef BatchNorm2BatchNorm3D::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  MS_EXCEPTION_IF_NULL(Xs);
  VectorRef pattern({prim::kPrimBatchNorm, Xs});
  return pattern;
}

const AnfNodePtr BatchNorm2BatchNorm3D::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode_bn = node->cast<CNodePtr>();
  if (!NeedFusion(graph, cnode_bn)) {
    return nullptr;
  }
  auto bn_3d = CreateBatchNorm3D(graph, cnode_bn);
  TransferDepend(cnode_bn, graph, bn_3d);
  return bn_3d;
}
}  // namespace opt
}  // namespace mindspore
