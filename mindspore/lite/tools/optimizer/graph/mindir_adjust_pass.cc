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
#include "tools/optimizer/graph/mindir_adjust_pass.h"
#include <algorithm>
#include <vector>
#include <memory>

#include "src/ops/primitive_c.h"
#include "tools/converter/converter_context.h"
#include "tools/converter/quantizer/quant_cast.h"
#include "src/common/log_adapter.h"
#include "src/tensor.h"

using mindspore::lite::PrimitiveC;
namespace mindspore {
namespace opt {
int MindirAdjustPass::ValueNodeInt64Convert(AnfNodePtr anf_node) {
  if (!utils::isa<ValueNodePtr>(anf_node)) {
    return lite::RET_NO_CHANGE;
  }
  auto valueNode = anf_node->cast<ValueNodePtr>();
  if (valueNode->abstract() == nullptr) {
    return lite::RET_NO_CHANGE;
  }
  auto abstractTensor = utils::cast<abstract::AbstractTensorPtr>(valueNode->abstract());
  if (abstractTensor == nullptr) {
    return lite::RET_NO_CHANGE;
  }
  auto value = abstractTensor->GetValueTrack();
  if (value != nullptr && value->isa<tensor::Tensor>()) {
    if (abstractTensor->element() == nullptr) {
      MS_LOG(ERROR) << "abstractTensor->element() is nullptr.";
      return RET_ERROR;
    }
    auto typePtr = abstractTensor->element()->GetTypeTrack();
    if (typePtr->type_id() == kNumberTypeInt64) {
      auto shape_vector = utils::cast<abstract::ShapePtr>(abstractTensor->BuildShape())->shape();
      auto dest_tensor_info = std::make_shared<tensor::Tensor>(kNumberTypeInt32, shape_vector);
      auto *dest_data_buf = reinterpret_cast<int32_t *>(dest_tensor_info->data_c());
      auto src_tensor_info = value->cast<tensor::TensorPtr>();
      auto *src_data_buf = reinterpret_cast<int64_t *>(src_tensor_info->data_c());
      MS_ASSERT(dest_tensor_info->ElementsNum() == src_tensor_info->ElementsNum());
      for (int i = 0; i < dest_tensor_info->ElementsNum(); i++) {
        dest_data_buf[i] = src_data_buf[i];
      }
      abstractTensor->set_value(dest_tensor_info);
      abstractTensor->set_type(TypeIdToType(kNumberTypeInt32));
      abstractTensor->element()->set_type(TypeIdToType(kNumberTypeInt32));
      valueNode->set_value(dest_tensor_info);
    }
  }
  return lite::RET_NO_CHANGE;
}

int MindirAdjustPass::ParameterNodeConvert(AnfNodePtr anf_node) {
  if (!utils::isa<ParameterPtr>(anf_node)) {
    MS_LOG(INFO) << "only parameter node need to convert tensor.";
    return lite::RET_NO_CHANGE;
  }
  auto param_node = anf_node->cast<ParameterPtr>();
  if (!param_node->has_default()) {
    MS_LOG(INFO) << "this is graph input, don't need to convert.";
    return lite::RET_NO_CHANGE;
  }
  if (utils::isa<ParamValueLitePtr>(param_node->default_param())) {
    MS_LOG(INFO) << "the tensor has been a paramvalueLite.";
    return lite::RET_NO_CHANGE;
  }
  ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();
  if (param_value == nullptr) {
    MS_LOG(ERROR) << "fail to new a ParamValueLite.";
    return lite::RET_ERROR;
  }
  param_node->set_name(param_node->debug_info()->name());
  auto tensor_info = param_node->default_param()->cast<tensor::TensorPtr>();
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "the node is not a tensor::TensorPtr.";
    return lite::RET_ERROR;
  }
  param_value->set_tensor_size(tensor_info->Size());
  param_value->set_tensor_type(tensor_info->data_type());
  auto tensor_shape = tensor_info->shape();
  std::vector<int> shape;
  std::transform(tensor_shape.begin(), tensor_shape.end(), std::back_inserter(shape),
                 [](int64_t value) { return static_cast<int>(value); });
  param_value->set_tensor_shape(shape);
  auto *tensor = new (std::nothrow) lite::Tensor(tensor_info->data_type(), shape);
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "new a lite::tensor failed, get a nullptr.";
    return lite::RET_MEMORY_FAILED;
  }
  auto *tensor_data_buf = tensor->MutableData();
  if (tensor_data_buf == nullptr) {
    MS_LOG(ERROR) << "malloc tensor data failed.";
    delete tensor;
    return lite::RET_MEMORY_FAILED;
  }
  if (memcpy_s(tensor_data_buf, tensor_info->Size(), tensor_info->data_c(), tensor_info->Size()) != EOK) {
    MS_LOG(ERROR) << "memcpy_s error.";
    delete tensor;
    return lite::RET_MEMORY_FAILED;
  }
  tensor->set_data(nullptr);
  param_value->set_tensor_addr(tensor_data_buf);
  param_node->set_default_param(param_value);
  delete tensor;
  return lite::RET_OK;
}

int MindirAdjustPass::PrimitiveConvert(std::shared_ptr<AnfNode> anf_node) {
  if (!utils::isa<CNodePtr>(anf_node)) {
    MS_LOG(INFO) << "only cnode need to convert primitive.";
    return lite::RET_NO_CHANGE;
  }
  auto cnode = anf_node->cast<CNodePtr>();
  if (cnode->inputs().empty() || cnode->input(0) == nullptr) {
    MS_LOG(ERROR) << "the cnode is invalid.";
    return lite::RET_NULL_PTR;
  }
  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  if (value_node == nullptr || value_node->value() == nullptr) {
    MS_LOG(ERROR) << "value node is invalid.";
    return lite::RET_NULL_PTR;
  }
  if (utils::isa<PrimitiveCPtr>(value_node->value())) {
    MS_LOG(INFO) << "the value has been primitiveC.";
    return lite::RET_NO_CHANGE;
  }
  auto primitive = value_node->value()->cast<PrimitivePtr>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "the value is not primitive.";
    return lite::RET_ERROR;
  }
  auto inputs = cnode->inputs();
  inputs.erase(inputs.begin());
  if (!CheckPrimitiveType(anf_node, prim::kPrimReturn) && !CheckPrimitiveType(anf_node, prim::kPrimMakeTuple)) {
    auto primitive_c = PrimitiveC::Create(*primitive, inputs, quant_type_, train_flag_);
    if (primitive_c == nullptr) {
      MS_LOG(ERROR) << "fail to create a primitive_c: " << cnode->fullname_with_scope();
      lite::NoSupportOp::GetInstance()->InsertOp(primitive->name());
      return lite::RET_NOT_FIND_OP;
    }
    value_node->set_value(primitive_c);
  } else {
    auto primitiveT = std::make_unique<schema::PrimitiveT>();
    primitiveT->value.type = (CheckPrimitiveType(anf_node, prim::kPrimReturn) ? schema::PrimitiveType_Return
                                                                              : schema::PrimitiveType_MakeTuple);
    value_node->set_value(std::make_shared<PrimitiveC>(primitiveT.release()));
  }
  return lite::RET_OK;
}

bool MindirAdjustPass::Run(const FuncGraphPtr &graph) {
  if (this->fmk_type_ != lite::converter::FmkType_MS) {
    MS_LOG(INFO) << "The framework type of model should be mindir.";
    return lite::RET_OK;
  }
  MS_ASSERT(graph != nullptr);
  auto node_list = TopoSort(graph->get_return());
  int status = lite::RET_OK;
  bool success_flag = true;
  for (auto &node : node_list) {
    if (utils::isa<ParameterPtr>(node)) {
      status = ParameterNodeConvert(node);
    } else if (utils::isa<CNodePtr>(node)) {
      status = PrimitiveConvert(node);
    } else if (utils::isa<ValueNodePtr>(node)) {
      status = ValueNodeInt64Convert(node);
    }

    if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
      lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
      success_flag = false;
    }
  }
  if (!success_flag) {
    MS_LOG(ERROR) << "Adjust mindir failed.";
    return false;
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
