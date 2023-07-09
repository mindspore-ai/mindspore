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

#include "tools/converter/quantizer/split_shared_bias.h"
#include <set>
#include <string>
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/ccsrc/plugin/device/cpu/kernel/nnacl/op_base.h"
#include "tools/converter/quantizer/quant_params.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/lite_exporter/fetch_content.h"

namespace mindspore::lite::quant {
AnfNodePtr SplitSharedBias::CloneParameterNode(const CNodePtr &cnode, size_t index, const FuncGraphPtr &func_graph,
                                               const std::shared_ptr<ConverterPara> &param) {
  MS_ASSERT(cnode != nullptr && mirror_graph != nullptr);
  MS_CHECK_TRUE_RET(index < cnode->size(), nullptr);
  auto node = cnode->input(index);
  if (node == nullptr || utils::isa<mindspore::CNode>(node)) {
    MS_LOG(ERROR) << "this func cannot copy cnode.";
    return nullptr;
  }
  if (!utils::isa<Parameter>(node)) {
    MS_LOG(ERROR) << "this input node is not parameter_node.";
    return nullptr;
  }
  DataInfo data_info;
  STATUS status = FetchDataFromParameterNode(cnode, index, param->fmk_type, &data_info, true);
  if (status != RET_OK && status != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "fetch data failed.";
    return nullptr;
  }
  ShapeVector shape_vec(data_info.shape_.begin(), data_info.shape_.end());
  if (data_info.data_type_ == kObjectTypeTensorType) {
    shape_vec = ShapeVector{static_cast<int64_t>(data_info.data_.size() / sizeof(int))};
  }
  std::shared_ptr<tensor::Tensor> tensor_info;
  if (static_cast<TensorCompressionType>(data_info.compress_type_) == TensorCompressionType::kNoCompression) {
    tensor_info = std::make_shared<tensor::Tensor>(static_cast<TypeId>(data_info.data_type_), shape_vec);
  } else {
    tensor_info =
      std::make_shared<tensor::Tensor>(static_cast<TypeId>(data_info.data_type_), shape_vec, data_info.data_.size(),
                                       static_cast<TensorCompressionType>(data_info.compress_type_));
  }
  MS_CHECK_TRUE_RET(tensor_info != nullptr, nullptr);
  if (!data_info.data_.empty()) {
    auto tensor_data = reinterpret_cast<uint8_t *>(tensor_info->data_c());
    if (tensor_data == nullptr || tensor_info->data().nbytes() < 0) {
      MS_LOG(ERROR) << "tensor info data is nullptr or the size is smaller than zero.";
      return nullptr;
    }
    if (memcpy_s(tensor_data, tensor_info->data().nbytes(), data_info.data_.data(), data_info.data_.size()) != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed";
      return nullptr;
    }
  }
  tensor_info->set_quant_param(data_info.quant_params_);
  auto mirror_parameter = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(mirror_parameter != nullptr, nullptr);

  auto mirror_name = cnode->fullname_with_scope() + "_" + node->fullname_with_scope();
  MS_LOG(WARNING) << ">>>>>>> mirror name: " << mirror_name;
  mirror_parameter->set_name(mirror_name);
  mirror_parameter->set_default_param(tensor_info);
  mirror_parameter->set_abstract(tensor_info->ToAbstract());
  return mirror_parameter;
}

int SplitSharedBias::Run() {
  std::set<std::string> bias_names;
  auto manager = func_graph_->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, RET_ERROR, "manage is nullptr.");

  for (const auto &cnode : func_graph_->GetOrderedCnodes()) {
    if (!quant::CheckNodeInSet(cnode, kHasBiasOperator)) {
      continue;
    }
    if (cnode->inputs().size() <= (THIRD_INPUT + kPrimOffset)) {
      MS_LOG(WARNING) << "bias input not exist, cnode name: " << cnode->fullname_with_scope();
      continue;
    }
    auto input_node = cnode->input(THIRD_INPUT + kPrimOffset);
    CHECK_NULL_RETURN(input_node);
    if (input_node->isa<mindspore::Parameter>()) {
      auto bias_name = input_node->fullname_with_scope();
      auto ret = bias_names.insert(bias_name);
      if (ret.second) {
        continue;
      }
      // clone parameter_node
      auto mirro_input = CloneParameterNode(cnode, THIRD_INPUT + kPrimOffset, func_graph_, param_);
      if (mirro_input == nullptr) {
        MS_LOG(WARNING) << "clone bias node failed, cnode name: " << cnode->fullname_with_scope();
        return RET_ERROR;
      }
      manager->SetEdge(cnode, THIRD_INPUT + kPrimOffset, mirro_input);
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
