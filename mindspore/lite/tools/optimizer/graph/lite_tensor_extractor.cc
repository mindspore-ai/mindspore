/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/graph/lite_tensor_extractor.h"
#include <memory>
#include <vector>
#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/lite_ops.h"
#include "src/tensorlist.h"
#include "tools/optimizer/common/format_utils.h"
#include "utils/ms_utils_secure.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
namespace {
constexpr int kElementShapeIndex = 1;
constexpr int kElementNumOffset = 2;
constexpr int kBasicInfoMinSize = 3;
bool CheckTensorListIsValid(const std::vector<uint8_t> &tensorlist_data) {
  if (tensorlist_data.empty()) {
    return true;
  }
  auto basic_data_size = tensorlist_data.size() / sizeof(int);
  auto *data = reinterpret_cast<const int *>(tensorlist_data.data());
  if (basic_data_size < static_cast<size_t>(kBasicInfoMinSize)) {
    MS_LOG(ERROR) << "tensorlist data length illegal, which should be at least 3, now is " << basic_data_size;
    return false;
  }
  if (data[kElementShapeIndex] < 0 || INT_ADD_OVERFLOW(data[kElementShapeIndex], kBasicInfoMinSize)) {
    MS_LOG(ERROR) << "tensorlist data length is too big, INT add overflow.";
    return false;
  }
  if (static_cast<size_t>((data[kElementShapeIndex] + kBasicInfoMinSize)) > basic_data_size) {
    MS_LOG(ERROR) << "tensorlist data length illegal. current tensorlist data length should be at least "
                  << (data[kElementShapeIndex] + kBasicInfoMinSize) << ", but now is " << basic_data_size;
    return false;
  }
  auto element_num = data[data[kElementShapeIndex] + kElementNumOffset];
  if (element_num > 0 && INT_ADD_OVERFLOW(element_num, 1)) {
    MS_LOG(ERROR) << "tensorlist data length is too big, INT add overflow.";
    return false;
  }
  auto shape_once = data[kElementShapeIndex] + 1;
  auto shape_group_num = element_num < 0 ? 1 : element_num + 1;
  if (INT_MUL_OVERFLOW(shape_once, shape_group_num)) {
    MS_LOG(ERROR) << "tensorlist data length is too big, INT mul overflow.";
    return false;
  }
  auto shape_info_size = shape_once * shape_group_num;
  if (INT_ADD_OVERFLOW(shape_info_size, kElementNumOffset)) {
    MS_LOG(ERROR) << "tensorlist data length is too big, INT add overflow.";
    return false;
  }
  size_t real_data_size = static_cast<size_t>(shape_info_size + kElementNumOffset);
  if (real_data_size != basic_data_size) {
    MS_LOG(ERROR) << "current tensorlist data length should be " << real_data_size << ", but now is "
                  << basic_data_size;
    return false;
  }
  return true;
}

TensorPtr ConvertToLiteTensor(const lite::DataInfo &data_info) {
  auto tensor_category = lite::TensorCategory(lite::NodeType(data_info.node_type_), data_info.shape_.size(),
                                              TypeId(data_info.data_type_), data_info.data_.size());
  TensorPtr tensor;
  if (data_info.data_type_ != kObjectTypeTensorType) {
    tensor = std::make_shared<lite::Tensor>(TypeId(data_info.data_type_), data_info.shape_,
                                            (mindspore::Format)data_info.format_, tensor_category);
  } else {
    tensor = std::make_shared<lite::TensorList>(data_info.shape_, std::vector<int>(), tensor_category);
  }
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "new a lite tensor failed.";
    return nullptr;
  }
  auto tensor_size = data_info.data_.size();
  if (tensor_size > 0) {
    if (data_info.data_type_ == kObjectTypeTensorType) {
      auto tensor_list = std::static_pointer_cast<lite::TensorList>(tensor);
      if (!CheckTensorListIsValid(data_info.data_)) {
        MS_LOG(ERROR) << "tensor list is invalid.";
        return nullptr;
      }
      if (tensor_list->Decode(reinterpret_cast<const int *>(data_info.data_.data()), tensor_size) != RET_OK) {
        MS_LOG(ERROR) << "Decode tensorlist data failed.";
        return nullptr;
      }
    } else {
      auto tensor_data = malloc(tensor_size);
      if (tensor_data == nullptr) {
        MS_LOG(ERROR) << "tensor_data is nullptr.";
        return nullptr;
      }
      if (memcpy_s(tensor_data, tensor_size, data_info.data_.data(), tensor_size) != EOK) {
        free(tensor_data);
        MS_LOG(ERROR) << "memcpy data error.";
        return nullptr;
      }
      tensor->set_data(tensor_data);
    }
  }

  if (tensor_size == 0 && data_info.data_ptr_ != nullptr) {
    tensor->set_data(data_info.data_ptr_);
    tensor->set_own_data(false);
  }
  return tensor;
}

TensorPtr GetCNodeTensorListVarInput(const lite::DataInfo &data_info) {
  auto tensor_list = std::make_shared<lite::TensorList>(data_info.shape_, std::vector<int>{});
  if (tensor_list == nullptr) {
    MS_LOG(ERROR) << "new a lite tensor list failed";
    return nullptr;
  }
  if (data_info.data_.empty()) {
    return tensor_list;
  }
  if (!CheckTensorListIsValid(data_info.data_)) {
    MS_LOG(ERROR) << "tensor list is invalid.";
    return nullptr;
  }
  auto status = tensor_list->Decode(reinterpret_cast<const int *>(data_info.data_.data()), data_info.data_.size());
  if (status != RET_OK) {
    MS_LOG(ERROR) << "decode tensor list failed.";
    return nullptr;
  }
  return tensor_list;
}

TensorPtr CreateTensorFromData(const lite::DataInfo &data_info, const bool &has_inferred,
                               const mindspore::Format &format) {
  if (data_info.data_type_ == kObjectTypeTensorType) {
    auto tensor = GetCNodeTensorListVarInput(data_info);
    MS_CHECK_TRUE_MSG(tensor != nullptr, nullptr, "tensor is nullptr.");
    tensor->set_format((Format)(format));
    if (!has_inferred) {
      tensor->set_shape({-1});
    }
    return tensor;
  } else {
    auto tensor = std::make_shared<lite::Tensor>(TypeId(data_info.data_type_), data_info.shape_);
    MS_CHECK_TRUE_MSG(tensor != nullptr, nullptr, "tensor is nullptr.");
    tensor->set_format((Format)(format));
    if (!has_inferred) {
      tensor->set_shape({-1});
    }
    return tensor;
  }
}
}  // namespace

int LiteTensorExtractor::GetCNodeConstInputToAbstract(const CNodePtr &cnode, const AbstractBasePtrList &abs_list,
                                                      converter::FmkType fmk_type, bool train_flag) {
  MS_ASSERT(cnode != nullptr && const_ms_inputs != nullptr);
  for (size_t i = 1; i < cnode->size(); ++i) {
    if (utils::isa<CNodePtr>(cnode->input(i))) {
      continue;
    }
    STATUS status;
    lite::DataInfo data_info;
    if (utils::isa<ParameterPtr>(cnode->input(i))) {
      status = lite::FetchDataFromParameterNode(cnode, i, fmk_type, &data_info, true);
    } else {
      status = lite::FetchDataFromValueNode(cnode, i, fmk_type, train_flag, &data_info, true);
    }
    if (status == lite::RET_NO_CHANGE) {
      continue;
    }
    if (status != RET_OK) {
      MS_LOG(ERROR) << "fetch const input data failed.";
      return status;
    }

    auto abstract = abs_list[i - 1];
    if (abstract->isa<abstract::AbstractScalar>()) {
      continue;
    }

    if (!utils::isa<abstract::AbstractTensor>(abstract)) {
      if (utils::isa<abstract::AbstractScalar>(abstract)) {
        continue;
      }
      if (utils::isa<abstract::AbstractSequence>(abstract)) {
        continue;
      }
      MS_LOG(ERROR) << "abstract is not a AbstractTensor";
      return RET_ERROR;
    }
    auto shape_value = abstract->BuildValue();
    if (!shape_value->isa<tensor::Tensor>()) {
      if (SetAbstractTensorInfo(abstract) != RET_OK) {
        MS_LOG(ERROR) << "SetAbstractTensorInfo failed";
        return RET_ERROR;
      }
      shape_value = abstract->BuildValue();
    }
    auto input_tensor = shape_value->cast<tensor::TensorPtr>();
    MS_CHECK_FALSE(input_tensor == nullptr, RET_ERROR);
    if (input_tensor->data().const_data() != nullptr) {
      MS_LOG(DEBUG) << "abstract already have const data.";
      continue;
    }
    if (data_info.data_.size() == 0) {
      continue;
    }

    if (input_tensor->Size() > 0 && input_tensor->Size() == data_info.data_.size()) {
      if (EOK != common::huge_memcpy(reinterpret_cast<uint8_t *>(input_tensor->data_c()), input_tensor->Size(),
                                     data_info.data_.data(), data_info.data_.size())) {
        MS_LOG(ERROR) << "memcpy_s failed.";
        return RET_ERROR;
      }
    } else {
      MS_LOG(ERROR) << "the size of tensor data: {" << input_tensor->Size() << "} is not equal to the size of node: {"
                    << data_info.data_.size() << "}";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int LiteTensorExtractor::GetCNodeConstInputs(const CNodePtr &cnode, const converter::FmkType &fmk_type,
                                             const bool &train_flag, const bool &copy_data,
                                             std::vector<TensorPtr> *const_ms_inputs) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, RET_ERROR, "cnode is nullptr.");
  MS_CHECK_TRUE_MSG(const_ms_inputs != nullptr, RET_ERROR, "const_ms_inputs is nullptr.");
  for (size_t i = 1; i < cnode->size(); ++i) {
    if (utils::isa<CNodePtr>(cnode->input(i))) {
      continue;
    }
    if (GetCNodeConstInput(cnode, i, fmk_type, train_flag, copy_data, const_ms_inputs) != RET_OK) {
      MS_LOG(ERROR) << "get const inputs failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int LiteTensorExtractor::GetCNodeConstInput(const CNodePtr &cnode, const size_t &index,
                                            const converter::FmkType &fmk_type, const bool &train_flag,
                                            const bool &copy_data, std::vector<TensorPtr> *const_ms_inputs) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, RET_ERROR, "cnode is nullptr.");
  MS_CHECK_TRUE_MSG(const_ms_inputs != nullptr, RET_ERROR, "const_ms_inputs is nullptr.");
  if (utils::isa<CNodePtr>(cnode->input(index))) {
    return RET_OK;
  }
  STATUS status;
  lite::DataInfo data_info;
  if (utils::isa<ParameterPtr>(cnode->input(index))) {
    status = lite::FetchDataFromParameterNode(cnode, index, fmk_type, &data_info, copy_data);
  } else {
    status = lite::FetchDataFromValueNode(cnode, index, fmk_type, train_flag, &data_info, copy_data);
  }
  if (status == lite::RET_NO_CHANGE) {
    return RET_OK;
  }
  if (status != RET_OK) {
    MS_LOG(ERROR) << "fetch const input data failed.";
    return status;
  }
  auto tensor = ConvertToLiteTensor(data_info);
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "Create lite tensor from data info failed.";
    return RET_ERROR;
  }
  const_ms_inputs->push_back(tensor);
  return RET_OK;
}

int LiteTensorExtractor::GetCNodeVarInput(const CNodePtr &cnode, const size_t &index,
                                          const converter::FmkType &fmk_type, std::vector<TensorPtr> *var_ms_inputs) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, RET_ERROR, "cnode is nullptr.");
  MS_CHECK_TRUE_MSG(var_ms_inputs != nullptr, RET_ERROR, "var_ms_inputs is nullptr.");
  if (!utils::isa<CNodePtr>(cnode->input(index))) {
    MS_LOG(ERROR) << "The " << index << "th input for " << cnode->fullname_with_scope() << "should be cnode.";
    return RET_ERROR;
  }

  bool has_inferred{false};
  auto ret = DetermineCertainVarInputHasInferred(cnode, index, &has_inferred);
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "determine infer flag failed.");
  Format format{mindspore::NHWC};
  ret = opt::DetermineCertainVarInputFormat(cnode, index, &format);
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "determine format failed.");

  auto abstract = opt::GetCNodeInputAbstract(cnode, index);
  MS_CHECK_TRUE_MSG(abstract != nullptr, RET_ERROR, "abstract is nullptr.");
  if (utils::isa<abstract::AbstractTensor>(abstract)) {
    lite::DataInfo data_info;
    if (lite::FetchDataFromAbstract(abstract, &data_info) != RET_OK) {
      MS_LOG(ERROR) << "FetchDataFromAbstract failed.";
      return RET_ERROR;
    }
    auto tensor = CreateTensorFromData(data_info, has_inferred, format);
    MS_CHECK_TRUE_MSG(tensor != nullptr, RET_ERROR, "CreateTensorFromData failed.");
    var_ms_inputs->emplace_back(tensor);
  } else if (utils::isa<abstract::AbstractTuple>(abstract)) {
    auto tuple = std::reinterpret_pointer_cast<abstract::AbstractTuple>(abstract);
    MS_CHECK_TRUE_MSG(tuple != nullptr, RET_ERROR, "tuple is nullptr.");
    for (const auto &element : tuple->elements()) {
      lite::DataInfo data_info;
      if (lite::FetchDataFromAbstract(element, &data_info) != RET_OK) {
        MS_LOG(ERROR) << "FetchDataFromAbstract failed.";
        return RET_ERROR;
      }
      auto tensor = CreateTensorFromData(data_info, has_inferred, format);
      MS_CHECK_TRUE_MSG(tensor != nullptr, RET_ERROR, "CreateTensorFromData failed.");
      var_ms_inputs->emplace_back(tensor);
    }
  }
  return RET_OK;
}

int ModifyLiteDynamicShapeToOps(const AbstractBasePtr &abstract) {
  // change Lite dynamic shape {-1} to core/ops dynamic rank {-2}, will be removed after calling core/infer
  ShapeVector shape;
  if (opt::FetchShapeFromAbstract(abstract, &shape) != RET_OK) {
    MS_LOG(ERROR) << "FetchShapeFromAbstract failed.";
    return RET_ERROR;
  }
  if (shape.size() == 1 && shape[0] == -1) {
    auto dynamic_shape = std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    abstract->set_shape(dynamic_shape);
  }
  return RET_OK;
}

int LiteTensorExtractor::GetCNodeInputAbstractLists(const CNodePtr &cnode, AbstractBasePtrList *abs_list) {
  MS_ASSERT(cnode != nullptr);
  MS_ASSERT(abs_list != nullptr);
  auto origin_inputs = cnode->inputs();
  if (lite::RemoveIfDepend(cnode) != RET_OK) {
    MS_LOG(ERROR) << "remove depend failed.";
    cnode->set_inputs(origin_inputs);
    return RET_ERROR;
  }
  RemoveIfMonad(cnode);
  abs_list->clear();
  abs_list->reserve(cnode->size());
  for (size_t index = 1; index < cnode->size(); index++) {
    auto node = cnode->input(index);
    auto abs = node->abstract();
    if (abs == nullptr) {
      if (utils::isa<ValueNodePtr>(node)) {
        abs = node->cast<ValueNodePtr>()->value()->ToAbstract();
      } else {
        MS_LOG(ERROR) << "abstract is nullptr.";
        cnode->set_inputs(origin_inputs);
        return RET_ERROR;
      }
    }
    auto abstract = abs->Clone();
    if (abstract == nullptr) {
      MS_LOG(ERROR) << "CNode " << cnode->fullname_with_scope() << " get nullptr input abstract.";
      cnode->set_inputs(origin_inputs);
      return RET_ERROR;
    }

    if (utils::isa<abstract::AbstractTensor>(abstract)) {
      auto ret = ModifyLiteDynamicShapeToOps(abstract);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "ModifyLiteDynamicShapeToOps failed.";
        cnode->set_inputs(origin_inputs);
        return RET_ERROR;
      }
    } else if (utils::isa<abstract::AbstractTuple>(abstract)) {
      auto tuple = std::reinterpret_pointer_cast<abstract::AbstractTuple>(abstract);
      MS_CHECK_TRUE_MSG(tuple != nullptr, RET_ERROR, "tuple is nullptr.");
      for (const auto &element : tuple->elements()) {
        if (utils::isa<abstract::AbstractTensor>(element)) {
          auto ret = ModifyLiteDynamicShapeToOps(element);
          if (ret != RET_OK) {
            MS_LOG(ERROR) << "ModifyLiteDynamicShapeToOps failed.";
            cnode->set_inputs(origin_inputs);
            return RET_ERROR;
          }
        }
      }
    }
    abs_list->push_back(abstract);
  }
  cnode->set_inputs(origin_inputs);
  return RET_OK;
}

int LiteTensorExtractor::GetCNodeInputTensors(const CNodePtr &cnode, std::vector<TensorPtr> *inputs,
                                              converter::FmkType fmk_type, bool train_flag, bool copy_data) {
  MS_ASSERT(cnode != nullptr);
  MS_ASSERT(inputs != nullptr);
  auto origin_inputs = cnode->inputs();
  if (lite::RemoveIfDepend(cnode) != RET_OK) {
    MS_LOG(ERROR) << "remove depend failed.";
    return RET_ERROR;
  }
  RemoveIfMonad(cnode);

  for (size_t i = 1; i < cnode->size(); ++i) {
    if (utils::isa<CNodePtr>(cnode->input(i))) {
      std::vector<TensorPtr> var_inputs;
      if (GetCNodeVarInput(cnode, i, fmk_type, &var_inputs) != RET_OK) {
        MS_LOG(ERROR) << "get var inputs failed.";
        cnode->set_inputs(origin_inputs);
        return RET_ERROR;
      }
      inputs->insert(inputs->end(), var_inputs.begin(), var_inputs.end());
    } else {
      std::vector<TensorPtr> const_inputs;
      if (GetCNodeConstInput(cnode, i, fmk_type, train_flag, copy_data, &const_inputs) != RET_OK) {
        MS_LOG(ERROR) << "get const inputs failed.";
        cnode->set_inputs(origin_inputs);
        return RET_ERROR;
      }
      inputs->insert(inputs->end(), const_inputs.begin(), const_inputs.end());
    }
  }
  cnode->set_inputs(origin_inputs);
  return RET_OK;
}

int LiteTensorExtractor::GetCNodeOutputTensors(const CNodePtr &cnode, std::vector<TensorPtr> *outputs,
                                               bool train_flag) {
  MS_ASSERT(cnode != nullptr);
  MS_ASSERT(outputs != nullptr);
  std::vector<lite::DataInfo> data_infos;
  if (utils::isa<abstract::AbstractTuple>(cnode->abstract())) {
    auto tuple = std::reinterpret_pointer_cast<abstract::AbstractTuple>(cnode->abstract());
    if (tuple == nullptr) {
      MS_LOG(ERROR) << "tuple is nullptr.";
      return RET_ERROR;
    }
    auto elements = tuple->elements();
    for (size_t i = 0; i < elements.size(); i++) {
      lite::DataInfo data_info;
      data_info.node_type_ = lite::NodeType_CNode;
      if (train_flag) {
        data_infos.emplace_back(data_info);
        if (CheckPrimitiveType(cnode, prim::kPrimConv2DFusion) || CheckPrimitiveType(cnode, prim::kPrimAdam)) {
          break;
        }
      } else {
        if (!utils::isa<abstract::AbstractTensorPtr>(elements[i])) {
          MS_LOG(ERROR) << "abstract is not AbstractTensor.";
          return RET_ERROR;
        }
        auto type = kNumberTypeFloat32;
        if (utils::isa<abstract::AbstractTensorPtr>(elements[i])) {
          auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(elements[i]);
          MS_CHECK_TRUE_MSG(abstract_tensor != nullptr, RET_ERROR, "Cast to abstract tensor failed!");
          MS_CHECK_TRUE_RET(abstract_tensor->element() != nullptr, lite::RET_NULL_PTR);
          auto typePtr = abstract_tensor->element()->GetTypeTrack();
          MS_CHECK_TRUE_RET(typePtr != nullptr, lite::RET_NULL_PTR);
          type = typePtr->type_id();
        }
        data_info.data_type_ = type;
        data_infos.emplace_back(data_info);
        if (CheckPrimitiveType(cnode, prim::kPrimConv2DFusion)) {
          break;
        }
      }
    }
  } else {
    lite::DataInfo data_info;
    auto type = kNumberTypeFloat32;
    if (utils::isa<abstract::AbstractTensorPtr>(cnode->abstract())) {
      auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(cnode->abstract());
      MS_CHECK_TRUE_MSG(abstract_tensor != nullptr, RET_ERROR, "Cast to abstract tensor failed!");
      MS_CHECK_TRUE_RET(abstract_tensor->element() != nullptr, lite::RET_NULL_PTR);
      auto typePtr = abstract_tensor->element()->GetTypeTrack();
      MS_CHECK_TRUE_RET(typePtr != nullptr, lite::RET_NULL_PTR);
      type = typePtr->type_id();
    }
    data_info.data_type_ = type;
    data_info.node_type_ = lite::NodeType_CNode;
    data_infos.emplace_back(data_info);
  }
  for (const auto &data_info : data_infos) {
    auto tensor = ConvertToLiteTensor(data_info);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "Create lite tensor from data info failed.";
      return RET_ERROR;
    }
    outputs->push_back(tensor);
  }
  return RET_OK;
}
}  // namespace opt
}  // namespace mindspore
