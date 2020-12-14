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

#include "tools/anf_importer/import_from_meta_graphT.h"
#include <vector>
#include <algorithm>
#include "schema/inner/model_generated.h"
#include "frontend/operator/ops.h"
#include "src/param_value_lite.h"
#include "src/common/log_adapter.h"
#include "include/errorcode.h"

namespace mindspore::lite {
int AnfImporterFromMetaGraphT::ConverterConstTensor() {
  MS_ASSERT(nullptr != meta_graph_);
  MS_ASSERT(nullptr != func_graph_);
  for (size_t i = 0; i < meta_graph_->allTensors.size(); i++) {
    auto &tensor = meta_graph_->allTensors.at(i);
    MS_ASSERT(tensor != nullptr);
    if (tensor->nodeType != schema::NodeType::NodeType_ValueNode) {
      continue;
    }
    auto parameter = func_graph_->add_parameter();
    std::vector<int> shape(tensor->dims.size());
    std::copy(tensor->dims.begin(), tensor->dims.end(), shape.begin());
    auto type_id = static_cast<TypeId>(tensor->dataType);
    auto type_ptr = TypeIdToType(type_id);
    std::vector<int64_t> shape_vector;
    (void)std::transform(shape.begin(), shape.end(), std::back_inserter(shape_vector),
                         [](const int32_t &value) { return static_cast<int64_t>(value); });
    auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
    MS_ASSERT(nullptr != abstract_tensor);
    parameter->set_abstract(abstract_tensor);
    if (!tensor->name.empty()) {
      parameter->set_name(tensor->name);
    } else {
      parameter->set_name("const-" + std::to_string(i));
    }

    ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();
    MS_ASSERT(nullptr != param_value);
    param_value->set_tensor_shape(shape);
    param_value->set_tensor_type(type_id);
    param_value->set_format(tensor->format);
    if (!tensor->data.empty()) {
      auto size = tensor->data.size();
      char *tensor_data = new (std::nothrow) char[size];
      if (tensor_data == nullptr) {
        MS_LOG(ERROR) << "new char[] failed";
        return RET_MEMORY_FAILED;
      }
      auto ret = memcpy_s(tensor_data, size, tensor->data.data(), size);
      if (EOK != ret) {
        MS_LOG(ERROR) << "memcpy_s error";
        delete[] tensor_data;
        return RET_MEMORY_FAILED;
      }
      param_value->SetTensorData(tensor_data, size);
      parameter->set_default_param(param_value);
    } else if (std::find(meta_graph_->inputIndex.begin(), meta_graph_->inputIndex.end(), i) ==
               meta_graph_->inputIndex.end()) {
      parameter->set_default_param(param_value);
    }
    AddNode(i, parameter);
  }
  return RET_OK;
}

ValueNodePtr AnfImporterFromMetaGraphT::ConvertPrimitive(const std::unique_ptr<schema::CNodeT> &cNode) {
  MS_ASSERT(nullptr != meta_graph_);
  MS_ASSERT(nullptr != cNode);
  auto primitiveCValue = PrimitiveC::Create(cNode->primitive.release());
  if (primitiveCValue == nullptr) {
    MS_LOG(ERROR) << "fail to convert primitive";
    return nullptr;
  }
  cNode->primitive = nullptr;
  // add quant parameter
  for (auto index : cNode->inputIndex) {
    if (!meta_graph_->allTensors[index]->quantParams.empty()) {
      std::vector<schema::QuantParamT> quant_params(meta_graph_->allTensors[index]->quantParams.size());
      std::transform(
        meta_graph_->allTensors[index]->quantParams.begin(), meta_graph_->allTensors[index]->quantParams.end(),
        quant_params.begin(),
        [](std::unique_ptr<schema::QuantParamT> &quant_param) -> schema::QuantParamT { return *quant_param; });
      primitiveCValue->AddInputQuantParam(quant_params);
    } else {
      std::vector<schema::QuantParamT> notinited_quant_params(1);
      primitiveCValue->AddInputQuantParam(notinited_quant_params);
    }
  }
  for (auto index : cNode->outputIndex) {
    if (!meta_graph_->allTensors[index]->quantParams.empty()) {
      std::vector<schema::QuantParamT> quant_params(meta_graph_->allTensors[index]->quantParams.size());
      std::transform(
        meta_graph_->allTensors[index]->quantParams.begin(), meta_graph_->allTensors[index]->quantParams.end(),
        quant_params.begin(),
        [](std::unique_ptr<schema::QuantParamT> &quant_param) -> schema::QuantParamT { return *quant_param; });
      primitiveCValue->AddOutputQuantParam(quant_params);
    } else {
      std::vector<schema::QuantParamT> notinited_quant_params(1);
      primitiveCValue->AddOutputQuantParam(notinited_quant_params);
    }
  }
  auto value_node = NewValueNode(std::shared_ptr<PrimitiveC>(primitiveCValue));
  return value_node;
}

abstract::AbstractTensorPtr AnfImporterFromMetaGraphT::ConvertTensorToAbstractTensor(
  const std::unique_ptr<schema::TensorT> &tensor) {
  MS_ASSERT(nullptr != tensor);
  std::vector<int> shape(tensor->dims.size());
  std::copy(tensor->dims.begin(), tensor->dims.end(), shape.begin());
  auto type_id = static_cast<TypeId>(tensor->dataType);
  auto type_ptr = TypeIdToType(type_id);
  std::vector<int64_t> shape_vector;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(shape_vector),
                       [](const int32_t &value) { return static_cast<int64_t>(value); });
  auto ptr = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
  MS_ASSERT(nullptr != ptr);
  return ptr;
}

int AnfImporterFromMetaGraphT::ConvertAbstract(const std::unique_ptr<schema::CNodeT> &src_cnode,
                                               const CNodePtr &dst_cnode) {
  MS_ASSERT(nullptr != meta_graph_);
  MS_ASSERT(nullptr != src_cnode);
  MS_ASSERT(nullptr != dst_cnode);
  std::vector<uint32_t> out_tensor_ids = src_cnode->outputIndex;
  if (out_tensor_ids.size() == 1) {
    auto out_tensor_id = out_tensor_ids.front();
    MS_ASSERT(meta_graph_->allTensors.size() > out_tensor_id);
    auto &tensor = meta_graph_->allTensors.at(out_tensor_id);
    MS_ASSERT(nullptr != tensor);
    dst_cnode->set_abstract(ConvertTensorToAbstractTensor(tensor));
    AddNode(out_tensor_id, dst_cnode);
  } else {
    AbstractBasePtrList abstract_list;
    for (size_t i = 0; i < out_tensor_ids.size(); i++) {
      auto out_tensor_id = out_tensor_ids.at(i);
      MS_ASSERT(meta_graph_->allTensors.size() > out_tensor_id);
      auto &tensor = meta_graph_->allTensors.at(out_tensor_id);
      MS_ASSERT(nullptr != tensor);
      abstract_list.emplace_back(ConvertTensorToAbstractTensor(tensor));
      auto tuple_get_item_prim_ptr = GetTupleGetItemPrim();
      if (tuple_get_item_prim_ptr == nullptr) {
        MS_LOG(ERROR) << "GetTupleGetItemPrim return nullptr";
        return RET_NULL_PTR;
      }
      auto tuple_get_item_prim = NewValueNode(tuple_get_item_prim_ptr);
      auto get_item_value = NewValueNode(MakeValue<int>(i));
      if (tuple_get_item_prim == nullptr || get_item_value == nullptr) {
        MS_LOG(ERROR) << "NewValueNode is nullptr";
        return RET_NULL_PTR;
      }
      std::vector<AnfNodePtr> inputs{tuple_get_item_prim, dst_cnode, get_item_value};
      CNodePtr get_item_cnode = func_graph_->NewCNode(inputs);
      if (get_item_cnode == nullptr) {
        MS_LOG(ERROR) << "NewCNode is nullptr";
        return RET_NULL_PTR;
      }
      get_item_cnode->set_fullname_with_scope(src_cnode->name + "_getitem_" + std::to_string(i));
      AddNode(out_tensor_id, get_item_cnode);
    }
    dst_cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  }
  return RET_OK;
}

int AnfImporterFromMetaGraphT::ConverterCNode() {
  MS_ASSERT(nullptr != meta_graph_);
  MS_ASSERT(nullptr != func_graph_);
  for (const auto &cNode : meta_graph_->nodes) {
    MS_ASSERT(nullptr != cNode);
    auto anf_primitive = ConvertPrimitive(cNode);
    if (anf_primitive == nullptr) {
      MS_LOG(ERROR) << "cannot obtain anf primitive";
      return RET_NULL_PTR;
    }
    std::vector<AnfNodePtr> op_inputs = {anf_primitive};
    for (int j : cNode->inputIndex) {
      auto node = GetNode(j);
      if (nullptr == node) {
        MS_LOG(ERROR) << "Can't find input node.";
        return RET_NULL_PTR;
      }
      op_inputs.push_back(node);
    }
    auto new_cnode = func_graph_->NewCNode(op_inputs);
    MS_ASSERT(nullptr != new_cnode);
    new_cnode->set_fullname_with_scope(cNode->name);
    auto status = ConvertAbstract(cNode, new_cnode);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "ConvertAbstract failed.";
      return status;
    }
  }
  return RET_OK;
}

int AnfImporterFromMetaGraphT::AddReturnCNode() {
  MS_ASSERT(nullptr != meta_graph_);
  MS_ASSERT(nullptr != func_graph_);
  if (meta_graph_->outputIndex.size() > 1) {
    std::vector<AnfNodePtr> make_tuple_inputs;
    auto make_tuple_prim_ptr = GetMakeTuplePrim();
    if (make_tuple_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "GetMakeTuplePrim return nullptr";
      return RET_NULL_PTR;
    }
    auto make_tuple_prim = NewValueNode(make_tuple_prim_ptr);
    make_tuple_inputs.emplace_back(make_tuple_prim);
    for (auto tensor_id : meta_graph_->outputIndex) {
      auto cNode = GetNode(tensor_id);
      if (nullptr == cNode) {
        MS_LOG(ERROR) << "Can't find input node.";
        return RET_ERROR;
      }
      make_tuple_inputs.emplace_back(cNode);
    }
    auto make_tuple_cnode = func_graph_->NewCNode(make_tuple_inputs);
    if (make_tuple_cnode == nullptr) {
      MS_LOG(ERROR) << "NewCNode is nullptr";
      return RET_NULL_PTR;
    }
    make_tuple_cnode->set_fullname_with_scope("return tuple");

    std::vector<AnfNodePtr> op_inputs;
    auto return_prim_ptr = GetReturnPrim();
    if (return_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "GetReturnPrim return nullptr";
      return RET_NULL_PTR;
    }
    auto value_node = NewValueNode(return_prim_ptr);
    op_inputs.emplace_back(value_node);
    op_inputs.emplace_back(make_tuple_cnode);
    auto cnode = func_graph_->NewCNode(op_inputs);
    MS_ASSERT(nullptr != cnode);
    cnode->set_fullname_with_scope("return");
    func_graph_->set_return(cnode);
  } else {
    auto return_prim_ptr = GetReturnPrim();
    if (return_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "GetReturnPrim return nullptr";
      return RET_NULL_PTR;
    }
    auto value_node = NewValueNode(return_prim_ptr);
    std::vector<AnfNodePtr> op_inputs{value_node};
    auto cnode = GetNode(meta_graph_->outputIndex.front());
    if (nullptr == cnode) {
      MS_LOG(ERROR) << "Can't find input node.";
      return RET_ERROR;
    }
    op_inputs.emplace_back(cnode);
    auto return_cnode = func_graph_->NewCNode(op_inputs);
    if (return_cnode == nullptr) {
      MS_LOG(ERROR) << "NewCNode is nullptr";
      return RET_NULL_PTR;
    }
    return_cnode->set_fullname_with_scope("return");
    func_graph_->set_return(return_cnode);
  }
  return RET_OK;
}

FuncGraphPtr AnfImporterFromMetaGraphT::GetResult() { return this->func_graph_; }
}  // namespace mindspore::lite
