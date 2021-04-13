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

#include "tools/optimizer/graph/node_infershape.h"
#include <algorithm>
#include <memory>
#include <vector>
#include "tools/anf_exporter/anf_exporter.h"
#include "tools/common/node_util.h"
#include "tools/common/tensor_util.h"
#include "src/ops/populate/populate_register.h"
#include "src/ops/ops_utils.h"
#include "src/runtime/infer_manager.h"
#include "src/tensorlist.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t INITIAL_SIZE = 1024;
void FreeTensors(std::vector<lite::Tensor *> *tensors) {
  if (tensors == nullptr) {
    return;
  }
  for (auto &v : *tensors) {
    delete v;
    v = nullptr;
  }
  tensors->resize(0);
}

void SetConvWeightFormat(const CNodePtr &cnode, const std::vector<lite::Tensor *> &inputs) {
  MS_ASSERT(cnode != nullptr);
  if (!CheckPrimitiveType(cnode, prim::kPrimConv2DFusion) &&
      !CheckPrimitiveType(cnode, kPrimConv2DBackpropInputFusion) &&
      !CheckPrimitiveType(cnode, prim::kPrimConv2dTransposeFusion)) {
    return;
  }
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_ASSERT(prim != nullptr);
  if (prim->GetAttr(kWeightFormat) != nullptr && inputs.size() > 1) {
    inputs[1]->set_format(static_cast<schema::Format>(GetValue<int64_t>(prim->GetAttr(opt::kWeightFormat))));
  }
}

bool DuceInferFlag(const CNodePtr &cnode, const std::vector<lite::Tensor *> &inputs, FmkType fmk_type) {
  MS_ASSERT(cnode != nullptr);
  for (auto &input : inputs) {
    auto shape = input->shape();
    if (std::find(shape.begin(), shape.end(), -1) != shape.end()) {
      if (fmk_type == lite::converter::FmkType_ONNX && shape.size() == 4 && shape[3] == 3 && shape[1] == -1) {
        input->set_format(schema::Format_NHWC);
      }
      return false;
    }
  }
  auto origin_inputs = cnode->inputs();
  lite::AnfExporter::RemoveIfDepend(cnode);
  lite::AnfExporter::RemoveIfMakeTuple(cnode);
  for (size_t i = 1; i < cnode->size(); ++i) {
    if (!utils::isa<CNodePtr>(cnode->input(i))) {
      continue;
    }
    auto input_cnode = cnode->input(i)->cast<CNodePtr>();
    if (CheckPrimitiveType(cnode->input(i), prim::kPrimTupleGetItem)) {
      input_cnode = input_cnode->input(1)->cast<CNodePtr>();
    }
    if (input_cnode == nullptr) {
      MS_LOG(ERROR) << "input is not cnode.";
      cnode->set_inputs(origin_inputs);
      return false;
    }
    auto prim = GetValueNode<PrimitivePtr>(input_cnode->input(0));
    if (prim == nullptr || prim->GetAttr(kInferDone) == nullptr) {
      MS_LOG(ERROR) << "prim is invalid.";
      cnode->set_inputs(origin_inputs);
      return false;
    }
    if (!GetValue<bool>(prim->GetAttr(kInferDone))) {
      cnode->set_inputs(origin_inputs);
      return false;
    }
  }
  cnode->set_inputs(origin_inputs);
  return true;
}

tensor::TensorPtr NewTensorInfo(lite::Tensor *tensor) {
  std::vector<int> shape(tensor->shape());
  std::vector<int64_t> shape_vector(shape.begin(), shape.end());
  auto tensor_info = std::make_shared<tensor::Tensor>(tensor->data_type(), shape_vector);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "new tensor::Tensor failed";
    return nullptr;
  }
  return tensor_info;
}
}  // namespace

STATUS NodeInferShape::InferShape(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  auto anf_prim = GetValueNode<std::shared_ptr<Primitive>>(cnode->input(0));
  if (anf_prim == nullptr) {
    MS_LOG(DEBUG) << "primitive is nullptr";
    return lite::RET_ERROR;
  }
  anf_prim->AddAttr(kInferDone, MakeValue<bool>(false));
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  if (GetCNodeInputTensors(cnode, &inputs) != lite::RET_OK) {
    FreeTensors(&inputs);
    MS_LOG(ERROR) << "get inputs failed.";
    return lite::RET_ERROR;
  }
  SetConvWeightFormat(cnode, inputs);
  if (GetCNodeOutputTensors(cnode, &outputs) != lite::RET_OK) {
    FreeTensors(&inputs);
    FreeTensors(&outputs);
    MS_LOG(ERROR) << "get outputs failed.";
    return lite::RET_ERROR;
  }
  auto prim_t = lite::GetPrimitiveT(cnode->input(0));
  if (prim_t == nullptr) {
    MS_LOG(DEBUG) << "prim_t is nullptr";
    FreeTensors(&inputs);
    FreeTensors(&outputs);
    return lite::RET_ERROR;
  }
  flatbuffers::FlatBufferBuilder fbb(INITIAL_SIZE);
  auto prim = lite::ConvertToPrimitive(prim_t, &fbb);
  delete prim_t;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "get primitive failed.";
    FreeTensors(&inputs);
    FreeTensors(&outputs);
    fbb.Clear();
    return lite::RET_ERROR;
  }
  auto parameter_gen = lite::PopulateRegistry::GetInstance()->GetParameterCreator(prim->value_type(), lite::SCHEMA_CUR);
  if (parameter_gen == nullptr) {
    MS_LOG(ERROR) << "PopulateParameter return nullptr, type: " << schema::EnumNamePrimitiveType(prim->value_type());
    FreeTensors(&inputs);
    FreeTensors(&outputs);
    fbb.Clear();
    return lite::RET_ERROR;
  }
  auto parameter = parameter_gen(prim);
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "parameter is nullptr.";
    FreeTensors(&inputs);
    FreeTensors(&outputs);
    fbb.Clear();
    return lite::RET_ERROR;
  }
  parameter->infer_flag_ = DuceInferFlag(cnode, inputs, fmk_type_);
  auto status = KernelInferShape(inputs, &outputs, parameter);
  if (status == lite::RET_OK) {
    anf_prim->AddAttr(kInferDone, MakeValue<bool>(true));
  }
  if (status == lite::RET_OK || status == lite::RET_INFER_INVALID) {
    auto set_status = SetCNodeAbstract(cnode, outputs);
    if (set_status != lite::RET_OK) {
      MS_LOG(ERROR) << "set CNode abstract failed: " << cnode->fullname_with_scope();
      return set_status;
    }
  } else {
    MS_LOG(ERROR) << "infer shape failed.";
  }
  FreeTensors(&inputs);
  FreeTensors(&outputs);
  free(parameter);
  fbb.Clear();
  return status;
}

std::vector<int> NodeInferShape::GetInputShape(const CNodePtr &cnode, size_t index) {
  MS_ASSERT(cnode != nullptr);
  if (index >= cnode->size()) {
    return {};
  }
  auto origin_inputs = cnode->inputs();
  std::vector<AnfNodePtr> specify_inputs = {origin_inputs[0], origin_inputs[index]};
  cnode->set_inputs(specify_inputs);
  std::vector<lite::Tensor *> specify_tensors;
  if (GetCNodeInputTensors(cnode, &specify_tensors) != lite::RET_OK || specify_tensors.empty()) {
    cnode->set_inputs(origin_inputs);
    return {};
  }
  cnode->set_inputs(origin_inputs);
  auto shape = specify_tensors.front()->shape();
  FreeTensors(&specify_tensors);
  return shape;
}

std::vector<int> NodeInferShape::GetIntVecInput(const CNodePtr &cnode, size_t index) {
  MS_ASSERT(cnode != nullptr);
  if (index >= cnode->size()) {
    return {};
  }
  auto origin_inputs = cnode->inputs();
  std::vector<AnfNodePtr> specify_inputs = {origin_inputs[0], origin_inputs[index]};
  cnode->set_inputs(specify_inputs);
  std::vector<lite::Tensor *> specify_tensors;
  if (GetCNodeInputTensors(cnode, &specify_tensors) != lite::RET_OK || specify_tensors.empty()) {
    cnode->set_inputs(origin_inputs);
    return {};
  }
  cnode->set_inputs(origin_inputs);
  std::vector<int> tensor_data;
  if (specify_tensors.front()->data_type() != kNumberTypeInt32 &&
      specify_tensors.front()->data_type() != kNumberTypeInt) {
    FreeTensors(&specify_tensors);
    return {};
  }
  if (specify_tensors.front()->shape().size() != 1) {
    FreeTensors(&specify_tensors);
    return {};
  }
  tensor_data.resize(specify_tensors.front()->shape()[0]);
  if (memcpy_s(tensor_data.data(), tensor_data.size() * sizeof(int), specify_tensors.front()->data_c(),
               tensor_data.size() * sizeof(int)) != EOK) {
    FreeTensors(&specify_tensors);
    return {};
  }
  return tensor_data;
}

STATUS NodeInferShape::GetCNodeInputTensors(const CNodePtr &cnode, std::vector<lite::Tensor *> *inputs) {
  MS_ASSERT(cnode != nullptr);
  MS_ASSERT(inputs != nullptr);
  auto origin_inputs = cnode->inputs();
  lite::AnfExporter::RemoveIfDepend(cnode);
  lite::AnfExporter::RemoveIfMakeTuple(cnode);
  RemoveIfMonad(cnode);
  std::vector<lite::Tensor *> const_inputs;
  if (GetCNodeConstInput(cnode, &const_inputs) != lite::RET_OK) {
    MS_LOG(ERROR) << "get const inputs failed.";
    FreeTensors(&const_inputs);
    cnode->set_inputs(origin_inputs);
    return lite::RET_ERROR;
  }
  std::vector<lite::Tensor *> var_inputs;
  if (GetCNodeVarInput(cnode, &var_inputs) != lite::RET_OK) {
    MS_LOG(ERROR) << "get var inputs failed.";
    FreeTensors(&var_inputs);
    cnode->set_inputs(origin_inputs);
    return lite::RET_ERROR;
  }
  size_t const_index = 0;
  size_t var_index = 0;
  bool input_valid = true;
  for (size_t i = 1; i < cnode->size(); ++i) {
    if (utils::isa<CNodePtr>(cnode->input(i))) {
      if (var_index >= var_inputs.size()) {
        MS_LOG(ERROR) << "var inputs size invalid.";
        input_valid = false;
        break;
      }
      inputs->emplace_back(var_inputs[var_index++]);
    } else {
      if (const_index >= const_inputs.size()) {
        MS_LOG(ERROR) << "const inputs size invalid.";
        input_valid = false;
        break;
      }
      inputs->emplace_back(const_inputs[const_index++]);
    }
  }
  cnode->set_inputs(origin_inputs);
  if (!input_valid) {
    FreeTensors(&const_inputs);
    FreeTensors(&var_inputs);
    inputs->resize(0);
  }
  return lite::RET_OK;
}

STATUS NodeInferShape::GetCNodeConstInput(const CNodePtr &cnode, std::vector<lite::Tensor *> *const_ms_inputs) {
  MS_ASSERT(cnode != nullptr);
  auto origin_inputs = cnode->inputs();
  std::vector<AnfNodePtr> const_inputs;
  for (auto &input : origin_inputs) {
    if (utils::isa<CNodePtr>(input)) {
      continue;
    }
    const_inputs.push_back(input);
  }
  cnode->set_inputs(const_inputs);
  auto meta_graph = std::make_unique<schema::MetaGraphT>();
  meta_graph->fmkType = fmk_type_;
  auto fb_node = std::make_unique<schema::CNodeT>();
  lite::AnfExporter anf_exporter;
  anf_exporter.set_train_flag(train_flag_);
  auto status = anf_exporter.SetOpInputNode(cnode, meta_graph, fb_node.get());
  cnode->set_inputs(origin_inputs);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "get const inputs failed.";
    return status;
  }
  return ConvertToLiteTensor(meta_graph, fb_node->inputIndex, const_ms_inputs);
}

STATUS NodeInferShape::GetCNodeVarInput(const CNodePtr &cnode, std::vector<lite::Tensor *> *var_ms_inputs) {
  MS_ASSERT(cnode != nullptr);
  MS_ASSERT(var_ms_inputs != nullptr);
  for (size_t i = 1; i < cnode->size(); ++i) {
    if (!utils::isa<CNodePtr>(cnode->input(i))) {
      continue;
    }
    auto abstract = GetCNodeInputAbstract(cnode, i);
    if (abstract == nullptr) {
      MS_LOG(ERROR) << "Abstract cnode is nullptr.";
      return lite::RET_ERROR;
    }
    if (!utils::isa<abstract::AbstractTensorPtr>(abstract)) {
      MS_LOG(ERROR) << "Abstract should be anstract tensor.";
      return lite::RET_ERROR;
    }
    auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract);
    auto type_ptr = abstract_tensor->element()->GetTypeTrack();
    MS_ASSERT(typePtr != nullptr);
    if (!utils::isa<abstract::ShapePtr>(abstract_tensor->BuildShape())) {
      MS_LOG(ERROR) << "Shape of Abstract should be ShapePtr.";
      return lite::RET_ERROR;
    }
    auto shape_vector = utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape();
    std::vector<int32_t> dims(shape_vector.begin(), shape_vector.end());
    lite::Tensor *tensor = nullptr;
    if (type_ptr->type_id() == kObjectTypeTensorType) {
      tensor = GetCNodeTensorListVarInput(dims, abstract_tensor);
    } else {
      tensor = new (std::nothrow) lite::Tensor(TypeId(type_ptr->type_id()), dims);
    }
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "new a lite tensor failed";
      return lite::RET_ERROR;
    }
    var_ms_inputs->emplace_back(tensor);
  }
  return lite::RET_OK;
}

lite::Tensor *NodeInferShape::GetCNodeTensorListVarInput(std::vector<int> shape,
                                                         const abstract::AbstractTensorPtr &abstract_tensor) {
  MS_ASSERT(abstract_tensor != nullptr);
  auto tensor_list = new (std::nothrow) lite::TensorList(shape, {});
  if (tensor_list == nullptr) {
    MS_LOG(ERROR) << "new a lite tensor list failed";
    return nullptr;
  }
  auto tensor_info = abstract_tensor->GetValueTrack();
  if (tensor_info == nullptr || !utils::isa<tensor::TensorPtr>(tensor_info)) {
    delete tensor_list;
    MS_LOG(ERROR) << "nsor list abstract is invalid.";
    return nullptr;
  }
  auto tensor_value = tensor_info->cast<tensor::TensorPtr>();
  if (tensor_value->data_c() == nullptr) {
    delete tensor_list;
    MS_LOG(ERROR) << "cannot get tensor list abstract's info.";
    return nullptr;
  }
  auto status = tensor_list->Decode(static_cast<int *>(tensor_value->data_c()));
  if (status != lite::RET_OK) {
    delete tensor_list;
    MS_LOG(ERROR) << "decode tensor list failed.";
    return nullptr;
  }
  return tensor_list;
}

STATUS NodeInferShape::GetCNodeOutputTensors(const CNodePtr &cnode, std::vector<lite::Tensor *> *outputs) {
  MS_ASSERT(cnode != nullptr);
  MS_ASSERT(outputs != nullptr);
  auto meta_graph = std::make_unique<schema::MetaGraphT>();
  meta_graph->fmkType = fmk_type_;
  auto fb_node = std::make_unique<schema::CNodeT>();
  lite::AnfExporter anf_exporter;
  anf_exporter.set_train_flag(train_flag_);
  anf_exporter.SetOpOutputNode(cnode, meta_graph, fb_node.get());
  return ConvertToLiteTensor(meta_graph, fb_node->outputIndex, outputs);
}

STATUS NodeInferShape::ConvertToLiteTensor(const std::unique_ptr<schema::MetaGraphT> &meta_graph,
                                           const std::vector<uint32_t> &tensor_indexes,
                                           std::vector<lite::Tensor *> *tensors) {
  MS_ASSERT(meta_graph != nullptr);
  MS_ASSERT(tensors != nullptr);
  for (auto index : tensor_indexes) {
    auto tensor_t = meta_graph->allTensors.at(index).get();
    auto tensor_shape = tensor_t->dims;
    auto tensor_category = lite::TensorCategory(tensor_t->nodeType, tensor_t->dims.size(), TypeId(tensor_t->dataType),
                                                tensor_t->data.size());
    lite::Tensor *tensor = nullptr;
    if (tensor_t->dataType != kObjectTypeTensorType) {
      tensor =
        new (std::nothrow) lite::Tensor(TypeId(tensor_t->dataType), tensor_shape, tensor_t->format, tensor_category);
    } else {
      tensor = new (std::nothrow) lite::TensorList(tensor_shape, std::vector<int>(), tensor_category);
    }
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "new a lite tensor failed";
      return lite::RET_ERROR;
    }
    auto tensor_size = tensor_t->data.size() * sizeof(char);
    if (tensor_size > 0) {
      if (tensor_t->dataType == kObjectTypeTensorType) {
        auto tensor_list = reinterpret_cast<lite::TensorList *>(tensor);
        if (tensor_list->Decode(reinterpret_cast<const int *>(tensor_t->data.data())) != RET_OK) {
          MS_LOG(ERROR) << "Decode tensorlist data failed";
          return RET_ERROR;
        }
      } else {
        auto tensor_data = new (std::nothrow) char[tensor_size];
        if (tensor_data == nullptr) {
          MS_LOG(ERROR) << "tensor_data is nullptr";
          delete tensor;
          return lite::RET_ERROR;
        }
        if (memcpy_s(tensor_data, tensor_size, tensor_t->data.data(), tensor_size) != EOK) {
          delete tensor;
          delete[](tensor_data);
          MS_LOG(ERROR) << "memcpy error: ";
          return lite::RET_ERROR;
        }
        tensor->set_data(tensor_data);
      }
    }
    tensors->emplace_back(tensor);
  }
  return lite::RET_OK;
}

STATUS NodeInferShape::SetCNodeAbstract(const std::shared_ptr<CNode> &cnode,
                                        const std::vector<lite::Tensor *> &outputs) {
  MS_ASSERT(cnode != nullptr);
  if (outputs.size() == 0) {
    MS_LOG(ERROR) << "empty output_tensors";
    return RET_ERROR;
  }
  auto origin_abstract = cnode->abstract();
  if (outputs.size() == 1 && !utils::isa<abstract::AbstractTuple>(origin_abstract)) {
    auto tensor = outputs.front();
    auto new_abstract = ConvertLiteTensorToAbstract(tensor);
    if (new_abstract == nullptr) {
      return RET_ERROR;
    }
    cnode->set_abstract(new_abstract);
  } else {
    AbstractBasePtrList abstract_list;
    for (size_t i = 0; i < outputs.size(); i++) {
      auto tensor = outputs.at(i);
      auto new_abstract = ConvertLiteTensorToAbstract(tensor);
      if (new_abstract == nullptr) {
        return RET_ERROR;
      }
      abstract_list.emplace_back(new_abstract);
    }
    cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  }
  return RET_OK;
}

abstract::AbstractBasePtr NodeInferShape::ConvertLiteTensorToAbstract(lite::Tensor *tensor) {
  MS_ASSERT(nullptr != tensor);
  if (tensor->data_type() == kObjectTypeTensorType) {
    return ConvertTensorListToAbstract(tensor);
  }
  auto tensor_info = NewTensorInfo(tensor);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "new tensor::Tensor failed";
    return nullptr;
  }
  return tensor_info->ToAbstract();
}

// stract save tensorlist's type and shape. tensor_info save tensorlist's data and data type.
// both of them is different in term of shape and type.
abstract::AbstractBasePtr NodeInferShape::ConvertTensorListToAbstract(lite::Tensor *tensor) {
  MS_ASSERT(nullptr != tensor);
  auto tensor_list = dynamic_cast<lite::TensorList *>(tensor);
  if (tensor_list == nullptr) {
    MS_LOG(ERROR) << "cast tensor_list failed";
    return nullptr;
  }
  std::vector<int> shape(tensor->shape());
  std::vector<int64_t> shape_vector(shape.begin(), shape.end());
  auto tensor_list_abstract =
    std::make_shared<abstract::AbstractTensor>(TypeIdToType(tensor_list->data_type()), shape_vector);
  if (tensor_list_abstract == nullptr) {
    MS_LOG(ERROR) << "new AbstractTensor failed";
    return nullptr;
  }
  auto elememt_shape = tensor_list->element_shape();
  std::vector<int> data_info;
  data_info.push_back(tensor_list->tensors_data_type());
  data_info.push_back(elememt_shape.size());
  std::copy(elememt_shape.begin(), elememt_shape.end(), std::back_inserter(data_info));
  data_info.push_back(tensor_list->tensors().size());
  for (size_t i = 0; i < tensor_list->tensors().size(); ++i) {
    auto tensor_mem = tensor_list->tensors()[i];
    auto tensor_mem_shape = tensor_mem->shape();
    data_info.push_back(tensor_mem_shape.size());
    std::copy(tensor_mem_shape.begin(), tensor_mem_shape.end(), std::back_inserter(data_info));
  }
  std::vector<int64_t> data_shape;
  data_shape.push_back(data_info.size());
  auto tensor_info = std::make_shared<tensor::Tensor>(kNumberTypeInt32, data_shape, data_info.data(), kNumberTypeInt32);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "new tensor::Tensor failed";
    return nullptr;
  }
  tensor_list_abstract->set_value(tensor_info);
  return tensor_list_abstract;
}
}  // namespace opt
}  // namespace mindspore
