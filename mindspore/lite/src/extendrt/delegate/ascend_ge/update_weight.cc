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

#include "extendrt/delegate/ascend_ge/update_weight.h"
#include <string>
#include "ops/auto_generate/gen_lite_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "tools/common/string_util.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/core/ir/manager.h"
namespace mindspore {
namespace {
constexpr float kNumMicrosecondToMillisecond = 1000.0;
constexpr size_t kInputSize3 = 3;
constexpr size_t kConstantWeightShapeSize = 2;
constexpr size_t kInputIndex2 = 2;
constexpr const char *kUpdateWeightTensorNameSuffix = "_add_param";
constexpr const char *kUpdateWeightAddNodeNameSuffix = "_add_cnode";
constexpr std::size_t kUpdateWeightTensorNameSuffixSize = 10;
}  // namespace

bool UpdateWeight::IsMatchName(const std::string &cnode_name, const std::string &param_name) {
  if (find(constant_cnode_name_.begin(), constant_cnode_name_.end(), cnode_name) != constant_cnode_name_.end()) {
    MS_LOG(DEBUG) << "cnode name: " << cnode_name << ", param name: " << param_name;
    return true;
  }
  return false;
}

bool UpdateWeight::ParseUpdateWeightConfig(const std::string &names_str) {
  MS_LOG(DEBUG) << "names str: " << names_str;
  constant_cnode_name_ = mindspore::lite::SplitStringToVector(names_str, ',');
  if (constant_cnode_name_.empty()) {
    MS_LOG(ERROR) << "split name is empty, name str is: " << names_str;
    return false;
  }
  return true;
}

std::vector<std::string> UpdateWeight::GetVariableParamsName(const FuncGraphPtr &anf_graph) {
  return new_weight_param_name_;
}

bool UpdateWeight::SetInitDataNames(const std::vector<std::string> &init_data_names) {
  if (init_data_names.empty()) {
    MS_LOG(ERROR) << "init_data_names is empty.";
    return false;
  }
  init_data_names_ = init_data_names;
  return true;
}

bool UpdateWeight::UpdateConstantTensorData(const std::vector<std::vector<std::shared_ptr<tensor::Tensor>>> &weights,
                                            std::vector<std::vector<std::shared_ptr<tensor::Tensor>>> *new_weights) {
  // sort by init data name.
  if (new_weights == nullptr) {
    MS_LOG(ERROR) << "new_weight_tensors is nullptr.";
    return false;
  }
  auto time1 = lite::GetTimeUs();
  for (auto &weight : weights) {
    std::vector<std::shared_ptr<tensor::Tensor>> new_weight_tensors;
    std::map<std::string, std::shared_ptr<tensor::Tensor>> weights_pairs;
    for (auto tensor : weight) {
      MS_CHECK_TRUE_RET(tensor != nullptr, false);
      weights_pairs[tensor->name()] = tensor;
    }
    for (auto &init_data_name : init_data_names_) {
      auto size = init_data_name.size();
      if (size < kUpdateWeightTensorNameSuffixSize) {
        MS_LOG(ERROR) << "can not find init data name: " << init_data_name;
        return false;
      }
      auto name = init_data_name.substr(0, size - kUpdateWeightTensorNameSuffixSize);
      if (weights_pairs.find(name) == weights_pairs.end()) {
        MS_LOG(ERROR) << "can not find init data name in user update weight tensors.";
        return false;
      }
      auto weight_tensor = weights_pairs[name];
      weight_tensor->set_name(init_data_name);
      new_weight_tensors.push_back(weight_tensor);
    }
    new_weights->push_back(new_weight_tensors);
  }
  auto time2 = lite::GetTimeUs();
  MS_LOG(INFO) << "Calculate update tensor time: " << (time2 - time1) / kNumMicrosecondToMillisecond << " ms";
  return true;
}

bool UpdateWeight::CreateAddOpNodeForGraph(const FuncGraphPtr &anf_graph) {
  MS_CHECK_TRUE_RET(anf_graph != nullptr, false);
  if (constant_cnode_name_.empty()) {
    MS_LOG(ERROR) << "constant_cnode_name_ is empty, user not set config file for update weight.";
    return false;
  }
  auto node_list = TopoSort(anf_graph->get_return());
  for (auto &node : node_list) {
    MS_CHECK_TRUE_RET(node != nullptr, false);
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = utils::cast<CNodePtr>(node);
    MS_CHECK_TRUE_RET(cnode != nullptr, false);
    if (find(constant_cnode_name_.begin(), constant_cnode_name_.end(), cnode->fullname_with_scope()) ==
        constant_cnode_name_.end()) {
      continue;
    }
    if (cnode->size() < kInputSize3) {
      MS_LOG(ERROR) << "cnode input size less " << kInputSize3;
      return false;
    }
    auto weight = cnode->input(kInputIndex2);
    MS_CHECK_TRUE_RET(weight != nullptr, false);

    // create Add node
    auto add_prim = std::make_shared<ops::Add>();
    if (add_prim == nullptr) {
      MS_LOG(ERROR) << "create add prim failed.";
      return false;
    }
    auto add_prim_c = add_prim->GetPrim();
    MS_CHECK_TRUE_RET(add_prim_c != nullptr, false);
    if (!utils::isa<ParameterPtr>(weight)) {
      MS_LOG(ERROR) << "matmul weight is not constant, can not update weight.";
      return false;
    }
    auto weight_param = weight->cast<ParameterPtr>();
    MS_CHECK_TRUE_RET(weight_param != nullptr, false);
    auto value = weight_param->default_param();
    MS_CHECK_TRUE_RET(value != nullptr, false);
    auto weight_tensor = value->cast<std::shared_ptr<tensor::Tensor>>();
    MS_CHECK_TRUE_RET(weight_tensor != nullptr, false);
    auto weight_shape = weight_tensor->shape();
    if (weight_shape.size() != kConstantWeightShapeSize) {
      MS_LOG(ERROR) << "now only support 2 dims matmul constant weight.";
      return false;
    }
    std::vector<std::vector<float>> add_param_data;
    for (auto i = 0; i < weight_shape[0]; i++) {
      std::vector<float> data;
      for (auto j = 0; j < weight_shape[1]; j++) {
        data.push_back(0);
      }
      add_param_data.push_back(data);
    }
    AnfNodePtr add_param_node = opt::BuildFloatVec2DParameterNode(
      anf_graph, add_param_data, cnode->fullname_with_scope() + kUpdateWeightTensorNameSuffix);
    if (add_param_node == nullptr) {
      MS_LOG(ERROR) << "create param node failed.";
      return false;
    }
    new_weight_param_name_.push_back(cnode->fullname_with_scope() + "_add_param");
    auto inputs = {weight, add_param_node};
    auto add_cnode = anf_graph->NewCNode(add_prim_c, inputs);
    if (add_cnode == nullptr) {
      MS_LOG(ERROR) << "new add node failed.";
      return false;
    }
    add_cnode->set_fullname_with_scope(cnode->fullname_with_scope() + kUpdateWeightAddNodeNameSuffix);
    if (node->abstract() != nullptr) {
      add_cnode->set_abstract(node->abstract()->Clone());
    }
    auto manager = Manage(anf_graph);
    (void)manager->Replace(weight, add_cnode);
  }
  if (new_weight_param_name_.size() != constant_cnode_name_.size()) {
    MS_LOG(ERROR) << "init data name size is not equal user config file name size, new_weight_param_name_: "
                  << new_weight_param_name_.size() << ", constant_cnode_name_ size: " << constant_cnode_name_.size();
  }
  MS_LOG(INFO) << "new_weight_param_name_ size: " << new_weight_param_name_.size()
               << ", constant_cnode_name_ size: " << constant_cnode_name_.size();
  return true;
}
}  // namespace mindspore
