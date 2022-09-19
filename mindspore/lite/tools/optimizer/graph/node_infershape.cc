/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#define USE_DEPRECATED_API
#include "tools/optimizer/graph/node_infershape.h"
#include <memory>
#include <vector>
#include <algorithm>
#include "src/common/primitive_t_utils.h"
#include "tools/common/node_util.h"
#include "tools/common/tensor_util.h"
#include "src/common/utils.h"
#include "src/common/ops/populate/populate_register.h"
#include "src/common/ops/anf_utils.h"
#include "src/litert/infer_manager.h"
#include "src/tensorlist.h"
#include "src/registry/kernel_interface_registry.h"
#include "tools/optimizer/graph/lite_tensor_extractor.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace opt {
namespace {
constexpr int kInputChannal = 3;
constexpr size_t INITIAL_SIZE = 1024;
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

tensor::TensorPtr NewTensorInfo(const lite::Tensor *tensor) {
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
  (void)anf_prim->AddAttr(kInferDone, MakeValue<bool>(false));
  std::vector<TensorPtr> inputs_ptr;
  if (LiteTensorExtractor::GetCNodeInputTensors(cnode, &inputs_ptr, fmk_type_, train_flag_, false) != lite::RET_OK) {
    MS_LOG(ERROR) << "get inputs failed.";
    return lite::RET_ERROR;
  }
  std::vector<TensorPtr> outputs_ptr;
  if (LiteTensorExtractor::GetCNodeOutputTensors(cnode, &outputs_ptr, train_flag_) != lite::RET_OK) {
    MS_LOG(ERROR) << "get outputs failed.";
    return lite::RET_ERROR;
  }
  auto prim_t = lite::GetPrimitiveT(cnode->input(0));
  if (prim_t == nullptr) {
    MS_LOG(DEBUG) << "prim_t is nullptr";
    return lite::RET_ERROR;
  }
  flatbuffers::FlatBufferBuilder fbb(INITIAL_SIZE);
  auto prim = lite::ConvertToPrimitive(prim_t.get(), &fbb);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "get primitive failed.";
    fbb.Clear();
    return lite::RET_ERROR;
  }
  std::vector<lite::Tensor *> inputs;
  (void)std::transform(inputs_ptr.begin(), inputs_ptr.end(), std::back_inserter(inputs),
                       [](const TensorPtr &input) { return input.get(); });
  std::vector<lite::Tensor *> outputs;
  (void)std::transform(outputs_ptr.begin(), outputs_ptr.end(), std::back_inserter(outputs),
                       [](const TensorPtr &output) { return output.get(); });
  auto ret = KernelInferShape(inputs, outputs, prim, {}, lite::SCHEMA_CUR);
  if (ret == lite::RET_NOT_SUPPORT) {
    auto parameter_gen = lite::PopulateRegistry::GetInstance()->GetParameterCreator(
      static_cast<int>(prim->value_type()), lite::SCHEMA_CUR);
    if (parameter_gen == nullptr) {
      MS_LOG(ERROR) << "PopulateParameter return nullptr, type: " << schema::EnumNamePrimitiveType(prim->value_type());
      fbb.Clear();
      return lite::RET_ERROR;
    }
    auto parameter = parameter_gen(prim);
    if (parameter == nullptr) {
      MS_LOG(ERROR) << "parameter is nullptr.";
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
    (void)anf_prim->AddAttr(kInferDone, MakeValue<bool>(true));
  }
  if (ret == lite::RET_OK || ret == lite::RET_INFER_INVALID) {
    auto set_status = SetCNodeAbstract(cnode, outputs, ret);
    (void)anf_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(inputs[0]->format()));
    if (set_status != lite::RET_OK) {
      MS_LOG(ERROR) << "set CNode abstract failed: " << cnode->fullname_with_scope();
      return set_status;
    }
  } else {
    MS_LOG(WARNING) << "infer shape failed.";
  }
  std::vector<int64_t> outputs_format;
  (void)std::transform(outputs.begin(), outputs.end(), std::back_inserter(outputs_format),
                       [](const lite::Tensor *output) { return output->format(); });
  (void)anf_prim->AddAttr(kOutputsFormat, MakeValue(outputs_format));
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
    status = lite::FetchDataFromCNode(base_node, position, &data_info);
  } else if (utils::isa<Parameter>(base_node->input(position))) {
    status = lite::FetchDataFromParameterNode(base_node, position, fmk_type_, &data_info, false);
  } else if (utils::isa<ValueNodePtr>(base_node->input(position))) {
    status = lite::FetchDataFromValueNode(base_node, position, fmk_type_, train_flag_, &data_info, false);
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
  std::vector<TensorPtr> specify_tensors;
  if (LiteTensorExtractor::GetCNodeInputTensors(cnode, &specify_tensors, fmk_type_, train_flag_, false) !=
        lite::RET_OK ||
      specify_tensors.empty()) {
    cnode->set_inputs(origin_inputs);
    return {};
  }
  cnode->set_inputs(origin_inputs);
  std::vector<int> tensor_data;
  if (specify_tensors.front()->data_type() != kNumberTypeInt32 &&
      specify_tensors.front()->data_type() != kNumberTypeInt) {
    return {};
  }
  if (specify_tensors.front()->shape().size() != 1) {
    return {};
  }
  MS_CHECK_GE(specify_tensors.front()->shape()[0], 0, {});
  tensor_data.resize(static_cast<size_t>(specify_tensors.front()->shape()[0]));
  if (memcpy_s(tensor_data.data(), tensor_data.size() * sizeof(int), specify_tensors.front()->data(),
               specify_tensors.front()->Size()) != EOK) {
    return {};
  }
  return tensor_data;
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
