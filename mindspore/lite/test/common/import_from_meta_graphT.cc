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

#include <vector>
#include <algorithm>
#include "schema/inner/model_generated.h"
#include "frontend/operator/ops.h"
#include "src/param_value_lite.h"
#include "src/common/log_adapter.h"
#include "tools/converter/converter_context.h"
#include "include/errorcode.h"
#include "test/common/import_from_meta_graphT.h"
#include "ir/func_graph.h"

namespace mindspore::lite {
AnfNodePtr AnfImporterFromMetaGraphT::GetNode(int tensor_id) {
  auto n = nodes_.find(tensor_id);
  if (n == nodes_.end()) {
    return nullptr;
  }
  return n->second;
}

void AnfImporterFromMetaGraphT::AddNode(int tensor_id, AnfNodePtr node) { nodes_[tensor_id] = std::move(node); }

int AnfImporterFromMetaGraphT::ConverterConstTensor() {
  MS_ASSERT(nullptr != meta_graph_);
  MS_ASSERT(nullptr != func_graph_);
  for (size_t i = 0; i < meta_graph_->allTensors.size(); i++) {
    auto &tensor = meta_graph_->allTensors.at(i);
    MS_ASSERT(tensor != nullptr);
    if (tensor->nodeType != NodeType_ValueNode) {
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
  return nullptr;
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
  return RET_ERROR;
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

int AnfImporterFromMetaGraphT::AddReturnCNode() { return RET_ERROR; }

FuncGraphPtr AnfImporterFromMetaGraphT::Fb2Anf(schema::MetaGraphT *meta_graph) {
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "meta_graph is null";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_NULL_PTR);
    return nullptr;
  }
  AnfImporterFromMetaGraphT anfImporterFromMetaGraphT(meta_graph);
  auto ret = anfImporterFromMetaGraphT.ConverterConstTensor();
  if (RET_OK != ret) {
    MS_LOG(ERROR) << "ConverterConstTensor failed " << ret;
    return nullptr;
  }
  ret = anfImporterFromMetaGraphT.ConverterCNode();
  if (RET_OK != ret) {
    MS_LOG(ERROR) << "ConverterCNode failed " << ret;
    return nullptr;
  }
  ret = anfImporterFromMetaGraphT.AddReturnCNode();
  if (RET_OK != ret) {
    MS_LOG(ERROR) << "AddReturnCNode failed " << ret;
    return nullptr;
  }
  return anfImporterFromMetaGraphT.func_graph_;
}
}  // namespace mindspore::lite
