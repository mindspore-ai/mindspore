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

#include "tools/optimizer/fusion/conv_conv_fusion.h"
#include <memory>
#include <vector>
#include "tools/common/tensor_util.h"
#include "ops/fusion/conv2d_fusion.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"

namespace mindspore::opt {
namespace {
constexpr size_t kConvNoBiasLen = 3;
constexpr size_t kConvWithBiasLen = 4;
constexpr size_t kConvWeightIndex = 2;
constexpr size_t kConvBiasIndex = 3;
constexpr size_t kNHWC_DIMS = 4;
constexpr size_t kNHWC_HDim = 1;
constexpr size_t kNHWC_WDim = 2;
constexpr size_t kNHWC_CDim = 3;

bool IsCommonConvNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    auto anf_node = utils::cast<AnfNodePtr>(n);
    if (!CheckPrimitiveType(anf_node, prim::kPrimConv2DFusion)) {
      return false;
    }
    std::shared_ptr<ops::Conv2DFusion> conv = nullptr;
    if (utils::isa<CNodePtr>(anf_node)) {
      auto c_node = anf_node->cast<CNodePtr>();
      conv = GetValueNode<std::shared_ptr<ops::Conv2DFusion>>(c_node->input(0));
    } else if (utils::isa<ValueNodePtr>(anf_node)) {
      conv = GetValueNode<std::shared_ptr<ops::Conv2DFusion>>(anf_node);
    }
    if (conv == nullptr) {
      return false;
    }
    return conv->GetAttr(ops::kIsDepthWise) == nullptr || !GetValue<bool>(conv->GetAttr(ops::kIsDepthWise));
  }
  return false;
}
STATUS GenNewConvBias(const ParameterPtr &down_bias_node, const ParameterPtr &down_weight_node,
                      const ParameterPtr &up_bias_node, const ParameterPtr &new_bias_node) {
  if (down_weight_node == nullptr || up_bias_node == nullptr || new_bias_node == nullptr) {
    MS_LOG(ERROR) << "Input down_weight_node or up_bias_node or new_bias_node is nullptr";
    return RET_FAILED;
  }
  float *down_bias_data = nullptr;
  if (down_bias_node != nullptr) {
    auto down_bias_param = std::dynamic_pointer_cast<tensor::Tensor>(down_bias_node->default_param());
    MS_ASSERT(down_bias_param != nullptr);
    auto down_bias_shape = down_bias_param->shape();
    if (down_bias_shape.size() != 1) {
      MS_LOG(ERROR) << "cur conv_conv fusion only support scalar bias shape";
      return RET_FAILED;
    }
    MS_ASSERT(down_bias_param->data_c() != nullptr);
    down_bias_data = static_cast<float *>(down_bias_param->data_c());
  }
  auto up_bias_param = std::dynamic_pointer_cast<tensor::Tensor>(up_bias_node->default_param());
  MS_ASSERT(up_bias_param != nullptr);
  auto up_bias_shape = up_bias_param->shape();
  if (up_bias_shape.size() != 1) {
    MS_LOG(ERROR) << "cur conv_conv fusion only support scalar bias shape";
    return RET_FAILED;
  }
  auto down_weight_param = std::dynamic_pointer_cast<tensor::Tensor>(down_weight_node->default_param());
  MS_ASSERT(down_weight_param != nullptr && down_weight_param->data_c() != nullptr);
  auto down_weight_data = static_cast<float *>(down_weight_param->data_c());
  auto down_weight_shape = down_weight_param->shape();
  MS_ASSERT(up_bias_param->data_c() != nullptr);
  auto up_bias_data = static_cast<float *>(up_bias_param->data_c());
  int new_bias_size = down_weight_shape[0];
  auto tensor_info = lite::CreateTensorInfo(nullptr, 0, {new_bias_size}, up_bias_param->data_type());
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "create tensor info failed.";
    return RET_ERROR;
  }
  MS_ASSERT(tensor_info->data_c() != nullptr);
  auto new_bias_data = static_cast<float *>(tensor_info->data_c());
  if (memset_s(new_bias_data, tensor_info->Size(), 0, new_bias_size * sizeof(float)) != EOK) {
    MS_LOG(ERROR) << "memset_s failed";
    return RET_ERROR;
  }
  auto up_bias_size = up_bias_shape[0];
  for (int i = 0; i < new_bias_size; i++) {
    for (int j = 0; j < up_bias_size; j++) {
      new_bias_data[i] += up_bias_data[j] * down_weight_data[i * up_bias_size + j];
    }
    if (down_bias_node != nullptr) {
      new_bias_data[i] += down_bias_data[i];
    }
  }
  new_bias_node->set_name(down_weight_node->fullname_with_scope());
  auto status = lite::InitParameterFromTensorInfo(new_bias_node, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return RET_ERROR;
  }
  return RET_OK;
}
// up weight shape[cout0,h,w,cin0] down weight shape[cout1,1,1,cout0],new weight shape [cout1,h,w,cin0]
STATUS GenNewConvWeight(const ParameterPtr &down_weight_node, const ParameterPtr &up_weight_node,
                        const ParameterPtr &new_weight_node) {
  MS_ASSERT(down_weight_node != nullptr && up_weight_node != nullptr && new_weight_node != nullptr);
  auto down_weight_param = std::dynamic_pointer_cast<tensor::Tensor>(down_weight_node->default_param());
  MS_ASSERT(down_weight_param != nullptr);
  MS_ASSERT(down_weight_param->data_c() != nullptr);
  auto down_weight_shape = down_weight_param->shape();
  auto up_weight_param = std::dynamic_pointer_cast<tensor::Tensor>(up_weight_node->default_param());
  MS_ASSERT(up_weight_param != nullptr);
  MS_ASSERT(up_weight_param->data_c() != nullptr);
  auto up_weight_shape = up_weight_param->shape();
  MS_CHECK_TRUE_RET(up_weight_shape.size() == kInputSizeFour, lite::RET_ERROR);
  auto up_weight_data = static_cast<float *>(up_weight_param->data_c());
  auto down_weight_data = static_cast<float *>(down_weight_param->data_c());
  int cout0 = up_weight_shape[0];
  int cin0 = up_weight_shape[kNHWC_CDim];
  int cout1 = down_weight_shape[0];
  int window_size = up_weight_shape[kNHWC_WDim] * up_weight_shape[kNHWC_HDim];
  auto new_weight_shape = up_weight_shape;
  new_weight_shape[0] = down_weight_shape[0];
  auto tensor_info = lite::CreateTensorInfo(nullptr, 0, new_weight_shape, up_weight_param->data_type());
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "create tensor info failed.";
    return RET_ERROR;
  }
  MS_ASSERT(tensor_info->data_c() != nullptr);
  auto new_weight_data = static_cast<float *>(tensor_info->data_c());
  for (int i = 0; i < cout1; i++) {
    auto down_weight_base = i * cout0;
    auto new_weight_base = i * window_size * cin0;
    for (int j = 0; j < cin0; j++) {
      for (int k = 0; k < cout0; k++) {
        auto up_weight_offset = k * window_size * cin0 + j;
        auto down_weight_offset = down_weight_base + k;

        auto new_weight_offset = new_weight_base + j;
        for (int m = 0; m < window_size; m++) {
          new_weight_data[new_weight_offset + cin0 * m] +=
            up_weight_data[up_weight_offset + cin0 * m] * down_weight_data[down_weight_offset];
        }
      }
    }
  }

  new_weight_node->set_name(down_weight_node->fullname_with_scope());
  auto status = lite::InitParameterFromTensorInfo(new_weight_node, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS ReplaceParametersAndNodes(const FuncGraphPtr &func_graph, const CNodePtr &up_conv_cnode,
                                 const CNodePtr &down_conv_cnode) {
  MS_ASSERT(func_graph != nullptr && up_conv_cnode != nullptr && down_conv_cnode != nullptr);
  auto down_weight_parameter = down_conv_cnode->input(kConvWeightIndex)->cast<ParameterPtr>();
  auto up_weight_parameter = up_conv_cnode->input(kConvWeightIndex)->cast<ParameterPtr>();
  auto new_weight_paramter = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(new_weight_paramter != nullptr, lite::RET_ERROR);
  if (GenNewConvWeight(down_weight_parameter, up_weight_parameter, new_weight_paramter) != RET_OK) {
    MS_LOG(ERROR) << "GenNewConvWeight failed.";
    return lite::RET_ERROR;
  }

  // whether up conv node has bias
  ParameterPtr new_bias_parameter{nullptr};
  if (up_conv_cnode->inputs().size() == kConvWithBiasLen) {
    ParameterPtr down_bias_parameter;
    if (down_conv_cnode->inputs().size() == kConvWithBiasLen) {
      down_bias_parameter = down_conv_cnode->input(kConvBiasIndex)->cast<ParameterPtr>();
    }
    auto up_bias_parameter = up_conv_cnode->input(kConvBiasIndex)->cast<ParameterPtr>();
    new_bias_parameter = func_graph->add_parameter();
    MS_CHECK_TRUE_RET(new_bias_parameter != nullptr, lite::RET_ERROR);
    if (GenNewConvBias(down_bias_parameter, down_weight_parameter, up_bias_parameter, new_bias_parameter) != RET_OK) {
      MS_LOG(ERROR) << "GenNewConvBias failed.";
      return lite::RET_ERROR;
    }
  }

  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  manager->SetEdge(down_conv_cnode, kConvWeightIndex, new_weight_paramter);
  if (new_bias_parameter != nullptr) {
    if (down_conv_cnode->inputs().size() == kConvWithBiasLen) {
      manager->SetEdge(down_conv_cnode, kConvBiasIndex, new_bias_parameter);
    } else {
      manager->AddEdge(down_conv_cnode, new_bias_parameter);
    }
  } else {
    MS_LOG(INFO) << "up conv node has no bias,no need to replace bias.";
  }
  MS_LOG(INFO) << "fusion node success:" << down_conv_cnode->fullname_with_scope();
  // delete up conv node
  (void)manager->Replace(up_conv_cnode, up_conv_cnode->input(1));
  return lite::RET_OK;
}

bool IsPrimitiveProper(const CNodePtr &up_conv_cnode, const CNodePtr &down_conv_cnode) {
  MS_ASSERT(up_conv_cnode != nullptr && down_conv_cnode != nullptr);
  auto down_conv_primitive = GetValueNode<std::shared_ptr<ops::Conv2DFusion>>(down_conv_cnode->input(0));
  MS_ASSERT(down_conv_primitive != nullptr);
  auto up_conv_primitive = GetValueNode<std::shared_ptr<ops::Conv2DFusion>>(up_conv_cnode->input(0));
  MS_ASSERT(up_conv_primitive != nullptr);
  int64_t up_conv_group = up_conv_primitive->GetAttr(ops::kGroup) == nullptr ? 1 : up_conv_primitive->get_group();
  int64_t down_conv_group = down_conv_primitive->GetAttr(ops::kGroup) == nullptr ? 1 : down_conv_primitive->get_group();
  int64_t up_pad_mode = up_conv_primitive->GetAttr(ops::kPadMode) == nullptr ? 0 : up_conv_primitive->get_pad_mode();
  int64_t down_pad_mode =
    down_conv_primitive->GetAttr(ops::kPadMode) == nullptr ? 0 : down_conv_primitive->get_pad_mode();
  return (up_conv_primitive->GetAttr(ops::kActivationType) == nullptr ||
          up_conv_primitive->get_activation_type() == mindspore::NO_ACTIVATION) &&
         up_conv_group == 1 && down_conv_group == 1 && up_pad_mode == down_pad_mode;
}
}  // namespace

const BaseRef ConvConvFusion::DefinePattern() const {
  auto is_conv_down = std::make_shared<CondVar>(IsCommonConvNode);
  MS_CHECK_TRUE_RET(is_conv_down != nullptr, {});
  auto is_conv_up = std::make_shared<CondVar>(IsCommonConvNode);
  MS_CHECK_TRUE_RET(is_conv_up != nullptr, {});
  auto is_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param != nullptr, {});
  auto is_seq_var = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var != nullptr, {});
  return VectorRef({is_conv_down, is_conv_up, is_param, is_seq_var});
}

bool ConvConvFusion::CheckCanFusion(const CNodePtr &up_conv_cnode, const CNodePtr &down_conv_cnode) const {
  if (IsMarkedTrainOp(down_conv_cnode)) {
    return false;
  }
  auto down_weight_parameter = down_conv_cnode->input(kConvWeightIndex)->cast<ParameterPtr>();
  MS_ASSERT(down_weight_parameter != nullptr);
  auto down_weight_value = std::dynamic_pointer_cast<tensor::Tensor>(down_weight_parameter->default_param());
  MS_ASSERT(down_weight_value != nullptr);
  auto down_weight_shape = down_weight_value->shape();
  auto down_weight_type = down_weight_value->data_type();
  // down conv node filter must 1x1,only support float32
  if (down_weight_shape.size() != kNHWC_DIMS || down_weight_type != kNumberTypeFloat32 ||
      (down_weight_shape[kNHWC_HDim] != 1 || down_weight_shape[kNHWC_WDim] != 1)) {
    return false;
  }
  if (IsMarkedTrainOp(up_conv_cnode)) {
    return false;
  }
  auto up_weight_parameter = up_conv_cnode->input(kConvWeightIndex)->cast<ParameterPtr>();
  MS_ASSERT(up_weight_parameter != nullptr);
  auto up_weight_value = std::dynamic_pointer_cast<tensor::Tensor>(up_weight_parameter->default_param());
  MS_ASSERT(up_weight_value != nullptr);
  auto up_weight_shape = up_weight_value->shape();
  auto up_weight_type = up_weight_value->data_type();
  if (up_weight_shape.size() != kNHWC_DIMS || up_weight_type != kNumberTypeFloat32 ||
      (up_weight_shape[kNHWC_HDim] != 1 || up_weight_shape[kNHWC_WDim] != 1)) {
    return false;
  }
  auto cin0 = up_weight_shape[kNHWC_CDim];
  auto cout0 = up_weight_shape[0];
  auto cout1 = down_weight_shape[0];
  if (cout0 != down_weight_shape[kNHWC_CDim]) {
    MS_LOG(WARNING) << "conv_conv_fusion up conv and down conv node shape not fit";
    return false;
  }
  if (cin0 * (cout1 - cout0) > cout0 * cout1) {
    MS_LOG(INFO) << "conv_conv_fusion up conv and down conv node channel requirement not fit";
    return false;
  }
  return true;
}

// conv->conv1x1 fusion conv (w1x+b)w2+c = (w1*w2)*x+(w2*b+c)
const AnfNodePtr ConvConvFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                         const EquivPtr &) const {
  if (func_graph == nullptr || node == nullptr) {
    return nullptr;
  }
  auto down_conv_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(down_conv_cnode != nullptr, nullptr);
  if (down_conv_cnode->inputs().size() != kConvWithBiasLen && down_conv_cnode->inputs().size() != kConvNoBiasLen) {
    MS_LOG(WARNING) << "conv node inputs error ,name:" << down_conv_cnode->fullname_with_scope();
    return nullptr;
  }
  auto up_conv_cnode = down_conv_cnode->input(1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(up_conv_cnode != nullptr, nullptr);
  if (up_conv_cnode->inputs().size() != kConvWithBiasLen && up_conv_cnode->inputs().size() != kConvNoBiasLen) {
    MS_LOG(WARNING) << "conv node inputs error ,name:" << up_conv_cnode->fullname_with_scope();
    return nullptr;
  }
  if (!CheckCanFusion(up_conv_cnode, down_conv_cnode)) {
    return nullptr;
  }
  // multi output need skip
  if (IsMultiOutputTensors(func_graph, up_conv_cnode)) {
    return nullptr;
  }
  // up conv node must no activation, and attributes should be proper
  if (!IsPrimitiveProper(up_conv_cnode, down_conv_cnode)) {
    return nullptr;
  }
  (void)ReplaceParametersAndNodes(func_graph, up_conv_cnode, down_conv_cnode);
  return nullptr;
}
}  // namespace mindspore::opt
