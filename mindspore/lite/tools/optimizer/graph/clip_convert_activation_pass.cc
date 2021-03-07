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
#include "tools/optimizer/graph/clip_convert_activation_pass.h"
#include <vector>
#include <memory>
#include "ops/clip.h"
#include "ops/fusion/activation.h"
#include "ops/op_utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/tensor.h"
#include "tools/converter/quantizer/quant_cast.h"
#include "src/common/log_adapter.h"

namespace mindspore::opt {
namespace {
constexpr size_t kClipMinIndex = 2;
constexpr size_t kClipMaxIndex = 3;
}  // namespace

bool ClipConvertActivationPass::Run(const FuncGraphPtr &graph) {
  MS_ASSERT(graph != nullptr);
  auto node_list = TopoSort(graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    if (!CheckPrimitiveType(node, prim::kPrimClip)) {
      continue;
    }
    auto clip_cnode = node->cast<CNodePtr>();
    MS_ASSERT(clip_cnode->size() >= kClipMinIndex);
    auto clip_c = GetValueNode<ops::PrimClipPtr>(clip_cnode->input(0));
    MS_ASSERT(clip_c != nullptr);
    float max = -1;
    float min = -1;
    if (clip_c->GetAttr(ops::kMax) != nullptr) {
      max = clip_c->get_max();
    }
    if (clip_c->GetAttr(ops::kMin) != nullptr) {
      min = clip_c->get_min();
    }
    if ((min == -1) && (max == -1)) {
      if (clip_cnode->size() > kClipMinIndex) {
        auto min_param_value = GetLiteParamValue(clip_cnode->input(kClipMinIndex));
        if (min_param_value->tensor_type() != mindspore::kNumberTypeFloat32) {
          MS_LOG(ERROR) << "Clip param type invalid";
          return false;
        }
        min = *reinterpret_cast<float *>(min_param_value->tensor_addr());
      } else {
        min = FLT_MIN;
      }

      if (clip_cnode->size() > kClipMaxIndex) {
        auto max_param_value = GetLiteParamValue(clip_cnode->input(kClipMaxIndex));
        if (max_param_value->tensor_type() != mindspore::kNumberTypeFloat32) {
          MS_LOG(ERROR) << "Clip param type invalid";
          return false;
        }
        max = *reinterpret_cast<float *>(max_param_value->tensor_addr());
      } else {
        max = FLT_MAX;
      }
    }
    auto manager = graph->manager();

    auto primitive_c = std::make_shared<mindspore::ops::Activation>();
    primitive_c->Init(0, min, max, mindspore::RELU6);
    if (min != 0 || max != 6) {
      primitive_c->set_activation_type(mindspore::HARD_TANH);
    }
    auto value_node = NewValueNode(primitive_c);
    std::vector<AnfNodePtr> op_inputs = {value_node};
    op_inputs.push_back(clip_cnode->input(1));
    auto new_cnode = graph->NewCNode(op_inputs);
    new_cnode->set_fullname_with_scope(node->fullname_with_scope());
    new_cnode->set_abstract(clip_cnode->abstract()->Clone());
    manager->Replace(node, new_cnode);
  }
  return false;
}
}  // namespace mindspore::opt
