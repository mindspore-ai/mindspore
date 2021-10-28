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
#include <memory>
#include <vector>
#include "tools/common/node_util.h"
#include "tools/common/tensor_util.h"
#include "src/common/utils.h"
#include "src/ops/populate/populate_register.h"
#include "src/ops/ops_utils.h"
#include "src/runtime/infer_manager.h"
#include "src/tensorlist.h"
#include "src/registry/kernel_interface_registry.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
namespace {
constexpr int kInputChannal = 3;
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

void RectifyFormat(const std::vector<lite::Tensor *> &inputs, FmkType fmk_type) {
  MS_ASSERT(cnode != nullptr);
  if (fmk_type != converter::kFmkTypeOnnx) {
    return;
  }
  for (auto &input : inputs) {
    auto shape = input->shape();
    if (shape.size() == kInputSizeFour && shape[kInputIndexThree] == kInputChannal && shape[1] == -1) {
      input->set_format(mindspore::NHWC);
    }
  }
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

bool NodeInferShape::JudgeOpSupportInfer(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  if (CheckPrimitiveType(cnode, prim::kPrimCustom)) {
    return true;
  }
  auto prim_t = lite::GetPrimitiveT(cnode->input(0));
  if (prim_t == nullptr) {
    return false;
  }
  auto parameter_gen =
    lite::PopulateRegistry::GetInstance()->GetParameterCreator(static_cast<int>(prim_t->value.type), lite::SCHEMA_CUR);
  if (parameter_gen == nullptr) {
    prim_t.reset();
    return false;
  }
  return true;
}

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
  auto prim = lite::ConvertToPrimitive(prim_t.get(), &fbb);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "get primitive failed.";
    FreeTensors(&inputs);
    FreeTensors(&outputs);
    fbb.Clear();
    return lite::RET_ERROR;
  }
  auto ret = KernelInferShape(inputs, outputs, prim, {}, lite::SCHEMA_CUR);
  if (ret == lite::RET_NOT_SUPPORT) {
    auto parameter_gen = lite::PopulateRegistry::GetInstance()->GetParameterCreator(
      static_cast<int>(prim->value_type()), lite::SCHEMA_CUR);
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
    RectifyFormat(inputs, fmk_type_);
    ret = KernelInferShape(inputs, outputs, parameter);
    if (parameter->destroy_func_ != nullptr) {
      parameter->destroy_func_(parameter);
    }
    free(parameter);
    parameter = nullptr;
  }
  fbb.Clear();
  if (ret == lite::RET_OK) {
    anf_prim->AddAttr(kInferDone, MakeValue<bool>(true));
  }
  if (ret == lite::RET_OK || ret == lite::RET_INFER_INVALID) {
    auto set_status = SetCNodeAbstract(cnode, outputs, ret);
    auto cnode_prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    MS_CHECK_TRUE_MSG(cnode_prim != nullptr, lite::RET_NULL_PTR, "GetValueNode Failed");
    cnode_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(inputs[0]->format()));
    if (set_status != lite::RET_OK) {
      MS_LOG(ERROR) << "set CNode abstract failed: " << cnode->fullname_with_scope();
      FreeTensors(&inputs);
      FreeTensors(&outputs);
      return set_status;
    }
  } else {
    MS_LOG(ERROR) << "infer shape failed.";
  }
  FreeTensors(&inputs);
  FreeTensors(&outputs);
  return ret;
}

std::vector<int> NodeInferShape::GetInputShape(const CNodePtr &cnode, size_t index) {
  MS_ASSERT(cnode != nullptr);
  if (index >= cnode->size()) {
    return {};
  }
  lite::DataInfo data_info;
  int status = lite::RET_OK;
  CNodePtr base_node = cnode;
  size_t position = index;
  if (CheckPrimitiveType(cnode->input(index), prim::kPrimMakeTuple) ||
      CheckPrimitiveType(cnode->input(index), kPrimMakeTupleV2)) {
    base_node = cnode->input(index)->cast<CNodePtr>();
    position = 1;
  }
  if (utils::isa<CNode>(base_node->input(position))) {
    status = lite::FetchDataFromCNode(base_node, position, fmk_type_, train_flag_, &data_info);
  } else if (utils::isa<Parameter>(base_node->input(position))) {
    status = lite::FetchDataFromParameterNode(base_node, position, fmk_type_, train_flag_, &data_info);
  } else if (utils::isa<ValueNodePtr>(base_node->input(position))) {
    status = lite::FetchDataFromValueNode(base_node, position, fmk_type_, train_flag_, &data_info);
  } else {
    MS_LOG(ERROR) << "input node is invalid.";
    return {};
  }
  if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
    MS_LOG(ERROR) << "fetch data failed.";
    return {};
  }
  return data_info.shape_;
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
  MS_CHECK_GE(specify_tensors.front()->shape()[0], 0, {});
  tensor_data.resize(static_cast<size_t>(specify_tensors.front()->shape()[0]));
  if (memcpy_s(tensor_data.data(), tensor_data.size() * sizeof(int), specify_tensors.front()->data(),
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
  lite::RemoveIfDepend(cnode);
  lite::RemoveIfMakeTuple(cnode);
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
  MS_ASSERT(cnode != nullptr && const_ms_inputs != nullptr);
  std::vector<lite::DataInfo> data_infos;
  for (size_t i = 1; i < cnode->size(); ++i) {
    if (utils::isa<CNodePtr>(cnode->input(i))) {
      continue;
    }
    STATUS status;
    lite::DataInfo data_info;
    if (utils::isa<ParameterPtr>(cnode->input(i))) {
      status = lite::FetchDataFromParameterNode(cnode, i, fmk_type_, train_flag_, &data_info);
    } else {
      status = lite::FetchDataFromValueNode(cnode, i, fmk_type_, train_flag_, &data_info);
    }
    if (status == lite::RET_NO_CHANGE) {
      continue;
    }
    if (status != lite::RET_OK) {
      MS_LOG(ERROR) << "fetch const input data failed.";
      return status;
    }
    data_infos.emplace_back(data_info);
  }
  return ConvertToLiteTensor(data_infos, const_ms_inputs);
}

STATUS NodeInferShape::GetCNodeVarInput(const CNodePtr &cnode, std::vector<lite::Tensor *> *var_ms_inputs) {
  MS_ASSERT(cnode != nullptr);
  MS_ASSERT(var_ms_inputs != nullptr);
  for (size_t i = 1; i < cnode->size(); ++i) {
    if (!utils::isa<CNodePtr>(cnode->input(i))) {
      continue;
    }
    lite::DataInfo data_info;
    if (lite::FetchDataFromCNode(cnode, i, fmk_type_, train_flag_, &data_info) != lite::RET_OK) {
      MS_LOG(ERROR) << "parse cnode failed.";
      return lite::RET_ERROR;
    }
    lite::Tensor *tensor = nullptr;
    if (data_info.data_type_ == kObjectTypeTensorType) {
      tensor = GetCNodeTensorListVarInput(data_info);
    } else {
      tensor = new (std::nothrow) lite::Tensor(TypeId(data_info.data_type_), data_info.shape_);
      tensor->set_format((Format)(data_info.format_));
    }
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "new a lite tensor failed";
      return lite::RET_ERROR;
    }
    auto input_cnode = cnode->input(i)->cast<CNodePtr>();
    MS_ASSERT(input_cnode != nullptr);
    PrimitivePtr input_prim = GetValueNode<PrimitivePtr>(input_cnode->input(0));
    if (CheckPrimitiveType(input_cnode, prim::kPrimTupleGetItem)) {
      auto item_input_cnode = input_cnode->input(1)->cast<CNodePtr>();
      MS_ASSERT(item_input_cnode != nullptr);
      input_prim = GetValueNode<PrimitivePtr>(item_input_cnode->input(0));
    }
    MS_ASSERT(input_prim != nullptr);
    if (input_prim->GetAttr(kInferDone) == nullptr || !GetValue<bool>(input_prim->GetAttr(kInferDone))) {
      tensor->set_shape({-1});
    }
    var_ms_inputs->emplace_back(tensor);
  }
  return lite::RET_OK;
}

lite::Tensor *NodeInferShape::GetCNodeTensorListVarInput(const lite::DataInfo &data_info) {
  auto tensor_list = new (std::nothrow) lite::TensorList(data_info.shape_, {});
  if (tensor_list == nullptr) {
    MS_LOG(ERROR) << "new a lite tensor list failed";
    return nullptr;
  }
  if (data_info.data_.empty()) {
    return tensor_list;
  }
  auto status = tensor_list->Decode(reinterpret_cast<const int *>(data_info.data_.data()));
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
  std::vector<lite::DataInfo> data_infos;
  if (utils::isa<abstract::AbstractTuple>(cnode->abstract())) {
    auto tuple = std::reinterpret_pointer_cast<abstract::AbstractTuple>(cnode->abstract());
    if (tuple == nullptr) {
      MS_LOG(ERROR) << "tuple is nullptr";
      return lite::RET_ERROR;
    }
    auto elements = tuple->elements();
    for (size_t i = 0; i < elements.size(); i++) {
      lite::DataInfo data_info;
      data_info.node_type_ = lite::NodeType_CNode;
      if (train_flag_) {
        data_infos.emplace_back(data_info);
        if (CheckPrimitiveType(cnode, prim::kPrimConv2DFusion) || CheckPrimitiveType(cnode, prim::kPrimAdam)) {
          break;
        }
      } else {
        if (!utils::isa<abstract::AbstractTensorPtr>(elements[i])) {
          MS_LOG(ERROR) << "abstract is not AbstractTensor";
          return lite::RET_ERROR;
        }
        auto type = kNumberTypeFloat32;
        if (utils::isa<abstract::AbstractTensorPtr>(elements[i])) {
          auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(elements[i]);
          auto typePtr = abstract_tensor->element()->GetTypeTrack();
          type = typePtr->type_id();
        }
        data_info.data_type_ = type;
        data_infos.emplace_back(data_info);
        if (CheckPrimitiveType(cnode, prim::kPrimConv2DFusion) ||
            CheckPrimitiveType(cnode, prim::kPrimFusedBatchNorm)) {
          break;
        }
      }
    }
  } else {
    lite::DataInfo data_info;
    auto type = kNumberTypeFloat32;
    if (utils::isa<abstract::AbstractTensorPtr>(cnode->abstract())) {
      auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(cnode->abstract());
      auto typePtr = abstract_tensor->element()->GetTypeTrack();
      type = typePtr->type_id();
    }
    data_info.data_type_ = type;
    data_info.node_type_ = lite::NodeType_CNode;
    data_infos.emplace_back(data_info);
  }
  return ConvertToLiteTensor(data_infos, outputs);
}

STATUS NodeInferShape::ConvertToLiteTensor(const std::vector<lite::DataInfo> &data_infos,
                                           std::vector<lite::Tensor *> *tensors) {
  MS_ASSERT(tensors != nullptr);
  for (auto &data_info : data_infos) {
    auto tensor_category = lite::TensorCategory(lite::NodeType(data_info.node_type_), data_info.shape_.size(),
                                                TypeId(data_info.data_type_), data_info.data_.size());
    lite::Tensor *tensor = nullptr;
    if (data_info.data_type_ != kObjectTypeTensorType) {
      tensor = new (std::nothrow) lite::Tensor(TypeId(data_info.data_type_), data_info.shape_,
                                               (mindspore::Format)data_info.format_, tensor_category);
    } else {
      tensor = new (std::nothrow) lite::TensorList(data_info.shape_, std::vector<int>(), tensor_category);
    }
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "new a lite tensor failed";
      return lite::RET_ERROR;
    }
    auto tensor_size = data_info.data_.size();
    if (tensor_size > 0) {
      if (data_info.data_type_ == kObjectTypeTensorType) {
        auto tensor_list = reinterpret_cast<lite::TensorList *>(tensor);
        if (tensor_list->Decode(reinterpret_cast<const int *>(data_info.data_.data())) != RET_OK) {
          MS_LOG(ERROR) << "Decode tensorlist data failed";
          return RET_ERROR;
        }
      } else {
        auto tensor_data = reinterpret_cast<char *>(malloc(tensor_size));
        if (tensor_data == nullptr) {
          MS_LOG(ERROR) << "tensor_data is nullptr";
          delete tensor;
          return lite::RET_ERROR;
        }
        if (memcpy_s(tensor_data, tensor_size, data_info.data_.data(), tensor_size) != EOK) {
          delete tensor;
          free(tensor_data);
          tensor_data = nullptr;
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

STATUS NodeInferShape::SetCNodeAbstract(const std::shared_ptr<CNode> &cnode, const std::vector<lite::Tensor *> &outputs,
                                        int status) {
  MS_ASSERT(cnode != nullptr);
  if (outputs.size() == 0) {
    MS_LOG(ERROR) << "empty output_tensors";
    return RET_ERROR;
  }
  auto origin_abstract = cnode->abstract();
  MS_ASSERT(origin_abstract != nullptr);
  if (outputs.size() == 1 && !utils::isa<abstract::AbstractTuple>(origin_abstract)) {
    auto tensor = outputs.front();
    auto new_abstract = ConvertLiteTensorToAbstract(tensor);
    if (new_abstract == nullptr) {
      MS_LOG(ERROR) << "new abstract failed.";
      return RET_ERROR;
    }
    if (status == lite::RET_INFER_INVALID) {
      ShapeVector shape;
      if (tensor->data_type() == kObjectTypeTensorType) {
        shape = {0};
      }
      auto abstract_shape = std::make_shared<abstract::Shape>(shape);
      CHECK_NULL_RETURN(abstract_shape);
      new_abstract->set_shape(abstract_shape);
    }
    cnode->set_abstract(new_abstract);
  } else {
    AbstractBasePtrList abstract_list;
    for (size_t i = 0; i < outputs.size(); i++) {
      auto tensor = outputs.at(i);
      auto new_abstract = ConvertLiteTensorToAbstract(tensor);
      if (new_abstract == nullptr) {
        MS_LOG(ERROR) << "new abstract failed.";
        return RET_ERROR;
      }
      if (status == lite::RET_INFER_INVALID) {
        ShapeVector shape;
        if (tensor->data_type() == kObjectTypeTensorType) {
          shape = {0};
        }
        auto abstract_shape = std::make_shared<abstract::Shape>(shape);
        CHECK_NULL_RETURN(abstract_shape);
        new_abstract->set_shape(abstract_shape);
      }
      abstract_list.emplace_back(new_abstract);
    }
    auto new_abstract_list = std::make_shared<abstract::AbstractTuple>(abstract_list);
    CHECK_NULL_RETURN(new_abstract_list);
    cnode->set_abstract(new_abstract_list);
  }
  return RET_OK;
}

abstract::AbstractBasePtr NodeInferShape::ConvertLiteTensorToAbstract(lite::Tensor *tensor) {
  MS_ASSERT(tensor != nullptr);
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
  MS_ASSERT(tensor != nullptr);
  auto tensor_list = reinterpret_cast<lite::TensorList *>(tensor);
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
