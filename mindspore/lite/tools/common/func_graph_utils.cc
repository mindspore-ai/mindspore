/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "tools/common/func_graph_utils.h"
#include <algorithm>
#include <memory>
#include "tools/common/graph_util.h"
#include "tools/converter/converter_context.h"
namespace mindspore {
AbstractBasePtr FuncGraphUtils::GetAbstractFromNode(const std::pair<AnfNodePtr, int64_t> &node) {
  auto anfnode = node.first;
  MS_EXCEPTION_IF_NULL(anfnode);
  AbstractBasePtr abstract = anfnode->abstract();
  if (abstract == nullptr) {
    return nullptr;
  }
  auto index = static_cast<size_t>(node.second);

  if (utils::isa<abstract::AbstractSequencePtr>(abstract)) {
    auto abstract_tuple = utils::cast<abstract::AbstractSequencePtr>(abstract);
    MS_EXCEPTION_IF_NULL(abstract_tuple);
    auto abstract_list = abstract_tuple->elements();
    if (abstract_list.size() <= index) {
      MS_LOG(WARNING) << "AbstractSequence's size[" << abstract_list.size() << "] is smaller than index " << index
                      << "]";
      return nullptr;
    }
    abstract = abstract_list[index];
  }
  return abstract;
}

std::string FuncGraphUtils::GetOutputName(const std::pair<AnfNodePtr, int64_t> &node_index) {
  auto node = node_index.first;
  auto idx = node_index.second;
  MS_EXCEPTION_IF_NULL(node);
  AbstractBasePtr abstract = GetAbstractFromNode(node_index);
  MS_EXCEPTION_IF_NULL(abstract);

  std::string output_name;
  if (!abstract->name().empty()) {
    output_name = abstract->name();
  } else if (idx >= 0) {
    output_name = node->fullname_with_scope() + "_" + std::to_string(idx);
  } else {
    output_name = node->fullname_with_scope();
  }

  return output_name;
}

void FuncGraphUtils::SetOutputName(const std::pair<AnfNodePtr, int64_t> &node, const std::string &name) {
  AbstractBasePtr abstract = GetAbstractFromNode(node);
  if (abstract != nullptr) {
    abstract->set_name(name);
  }
}

std::vector<std::string> FuncGraphUtils::GetFuncGraphOutputNames(const FuncGraphPtr &func_graph) {
  std::vector<std::string> output_names;
  // the 3rd model will save the tensor name to ConverterInnerContext
  output_names = lite::ConverterInnerContext::GetInstance()->GetGraphOutputTensorNames();
  if (!output_names.empty()) {
    return output_names;
  }
  std::vector<std::pair<AnfNodePtr, int64_t>> outputs;
  std::vector<std::string> tmp_names;
  std::vector<std::vector<int64_t>> tmp_dims;
  auto ret = lite::GetFuncGraphOutputsInfo(func_graph, &outputs, &tmp_names, &tmp_dims);
  MS_EXCEPTION_IF_CHECK_FAIL((ret == lite::RET_OK), "Get outputs info of funcgraph failed");

  output_names.resize(outputs.size());
  std::transform(outputs.begin(), outputs.end(), output_names.begin(), GetOutputName);
  return output_names;
}

void FuncGraphUtils::SetFuncGraphOutputNames(const FuncGraphPtr &func_graph,
                                             const std::vector<std::string> &output_names) {
  std::vector<std::pair<AnfNodePtr, int64_t>> outputs;
  std::vector<std::string> tmp_names;
  std::vector<std::vector<int64_t>> tmp_dims;
  auto ret = lite::GetFuncGraphOutputsInfo(func_graph, &outputs, &tmp_names, &tmp_dims);
  MS_EXCEPTION_IF_CHECK_FAIL((ret == lite::RET_OK), "Get outputs info of funcgraph failed");
  // the control flow model may be not equal, it will be updated by metagraph
  if (outputs.size() != output_names.size()) {
    MS_LOG(INFO)
      << "the size of output nodes is not equal to the size of output names, it will be updated by metagraph";
    return;
  }

  for (size_t i = 0; i < output_names.size(); ++i) {
    SetOutputName(outputs[i], output_names[i]);
  }
  return;
}

tensor::TensorPtr FuncGraphUtils::GetParameterConstValue(const AnfNodePtr &anf_node) {
  if (anf_node == nullptr) {
    MS_LOG(ERROR) << "Input argument anf node is nullptr";
    return nullptr;
  }
  auto parameter = anf_node->cast<ParameterPtr>();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "Node " << anf_node->fullname_with_scope() << " is not a Parameter";
    return nullptr;
  }
  auto default_param = parameter->default_param();
  if (default_param == nullptr) {
    MS_LOG(ERROR) << "Parameter " << anf_node->fullname_with_scope() << " has not default value";
    return nullptr;
  }
  if (!default_param->isa<tensor::Tensor>()) {
    MS_LOG(ERROR) << "Parameter " << anf_node->fullname_with_scope()
                  << " default value is not a tensor::Tensor, real type " << default_param->type_name();
    return nullptr;
  }
  auto tensor = default_param->cast<std::shared_ptr<tensor::Tensor>>();
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "Parameter " << anf_node->fullname_with_scope() << " tensor value is nullptr";
    return nullptr;
  }
  return tensor;
}
}  // namespace mindspore
