/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "tools/lite_exporter/fetch_content.h"
#include <algorithm>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <utility>
#include "tools/converter/quantizer/quant_param_holder.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_utils_secure.h"
#include "tools/optimizer/common/format_utils.h"
#include "tools/optimizer/graph/specify_graph_input_format.h"
#include "nnacl/op_base.h"
#include "tools/common/node_util.h"
#include "src/common/ops/anf_utils.h"
#include "src/common/ops/populate/populate_register.h"
#include "src/common/primitive_t_utils.h"
#include "mindapi/base/format.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace lite {
namespace {
constexpr int kNumWeightIndex = 2;
constexpr int kNumTransposePermSize = 4;
constexpr size_t kTensorListMinSize = 3 * sizeof(int32_t);
static const std::unordered_map<int, int> TypeToTypeMap = {
  {kNumberTypeInt, kNumberTypeInt32}, {kNumberTypeUInt, kNumberTypeUInt32}, {kNumberTypeFloat, kNumberTypeFloat32}};
STATUS GetShapeVectorFromStringTensor(const tensor::TensorPtr &tensor_info, ShapeVector *shape_vector, size_t *offset) {
  MS_ASSERT(tensor_info != nullptr && shape_vector != nullptr && offset != nullptr);
  auto data_type = tensor_info->data_type();
  if (data_type != kObjectTypeString) {
    MS_LOG(ERROR) << "This function only used for string tensor.";
    return RET_ERROR;
  }
  shape_vector->clear();
  MS_CHECK_TRUE_MSG(tensor_info->data_c() != nullptr, RET_ERROR, "tensor_info->data_c() is nullptr");
  auto tensor_data = reinterpret_cast<uint8_t *>(tensor_info->data_c());
  std::string shape_str;
  std::string shape_size_str;
  *offset = 0;
  size_t cnt = 0;
  for (; *offset < tensor_info->Size(); (*offset)++) {
    if (tensor_data[*offset] == ',') {
      (*offset)++;
      break;
    }
    shape_size_str.push_back(tensor_data[*offset]);
  }
  if (*offset == 0) {
    MS_LOG(ERROR) << "string tensor's dim size not found.";
    return RET_ERROR;
  }
  constexpr int kBase = 10;
  size_t shape_size = static_cast<size_t>(std::strtol(shape_size_str.c_str(), nullptr, kBase));
  MS_CHECK_TRUE_RET(shape_size != 0, RET_ERROR);
  for (; *offset < tensor_info->Size(); (*offset)++) {
    if (tensor_data[*offset] == ',') {
      cnt++;
      int64_t shape = 0;
      try {
        shape = std::stoi(shape_str);
      } catch (const std::exception &e) {
        MS_LOG(ERROR) << "Get shape failed: " << e.what();
        return RET_ERROR;
      }
      shape_vector->push_back(shape);
      shape_str.clear();
    } else {
      shape_str.push_back(tensor_data[*offset]);
    }
    if (cnt == shape_size) {
      (*offset)++;
      break;
    }
  }
  if (shape_vector->empty()) {
    MS_LOG(ERROR) << "string tensor's shape shouldn't be empty.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS GetDataTypeAndShape(const ParameterPtr &param_node, TypeId *data_type, ShapeVector *shape_vector) {
  MS_ASSERT(param_node != nullptr && data_type != nullptr && shape_vector != nullptr);
  auto abstract_base = param_node->abstract();
  if (abstract_base == nullptr) {
    MS_LOG(ERROR) << "Abstract of parameter is nullptr, " << param_node->name();
    return RET_PARAM_INVALID;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstract_base)) {
    MS_LOG(ERROR) << "Abstract of parameter should be anstract tensor, " << param_node->name();
    return RET_INPUT_TENSOR_ERROR;
  }
  auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract_base);
  MS_CHECK_TRUE_MSG(abstract_tensor != nullptr, RET_ERROR, "cast ptr failed");
  auto typePtr = abstract_tensor->element()->GetTypeTrack();
  MS_CHECK_TRUE_MSG(typePtr != nullptr, RET_ERROR, "typePtr is nullptr");
  *data_type = typePtr->type_id();
  if (!utils::isa<abstract::ShapePtr>(abstract_tensor->BuildShape())) {
    MS_LOG(ERROR) << "Shape of Abstract of parameter should be ShapePtr, " << param_node->name();
    return RET_PARAM_INVALID;
  }
  *shape_vector = utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape();
  return RET_OK;
}

int FetchFromTensorValue(const ValueNodePtr &value_node, converter::FmkType fmk_type, bool train_flag,
                         DataInfo *data_info, bool copy_data) {
  MS_ASSERT(value_node != nullptr && data_info != nullptr);
  auto valueAbstract = value_node->abstract();
  MS_CHECK_TRUE_MSG(valueAbstract != nullptr, RET_ERROR, "valueAbstract is nullptr");
  auto abstract_tensor = valueAbstract->cast<abstract::AbstractTensorPtr>();
  if (abstract_tensor == nullptr || abstract_tensor->element() == nullptr) {
    MS_LOG(ERROR) << "abstract_tensor or abstract_tensor->element() is nullptr";
    return RET_ERROR;
  }
  auto typePtr = abstract_tensor->element()->GetTypeTrack();
  MS_CHECK_TRUE_MSG(typePtr != nullptr, RET_ERROR, "typePtr is nullptr");
  data_info->data_type_ = typePtr->type_id();
  auto shape_vector = utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape();
  std::vector<int32_t> dims(shape_vector.begin(), shape_vector.end());
  data_info->shape_ = dims;
  if (train_flag && dims.empty()) {
    data_info->shape_ = {1};
  }
  auto value = value_node->value();
  MS_CHECK_TRUE_MSG(value != nullptr, RET_ERROR, "value is nullptr");
  auto data = value->cast<tensor::TensorPtr>();
  MS_CHECK_TRUE_MSG(data != nullptr, RET_ERROR, "data is invalid");
  if (data_info->format_ != mindspore::NHWC && data_info->format_ != mindspore::NCHW) {
    MS_LOG(ERROR) << "schema tensor format is wrong, " << data_info->format_;
    return RET_ERROR;
  }

  // process weight tensor
  if (copy_data) {
    data_info->data_.resize(data->Size());
    if (data->Size() > 0 && memcpy_s(data_info->data_.data(), data->Size(), data->data_c(), data->Size()) != EOK) {
      MS_LOG(ERROR) << "memcpy_s error.";
      return RET_ERROR;
    }
  } else {
    data_info->data_ptr_ = data->data_c();
  }
  return RET_OK;
}

template <typename DstImm, typename SrcImm>
int FetchCastImmValue(const ValueNodePtr &value_node, DataInfo *data_info) {
  MS_ASSERT(value_node != nullptr && data_info != nullptr);
  DstImm dst_imm;
  data_info->data_type_ = dst_imm.type()->number_type();
  data_info->shape_ = {1};
  data_info->data_.resize(sizeof(dst_imm.value()));
  auto value = value_node->value();
  MS_CHECK_TRUE_MSG(value != nullptr, RET_ERROR, "value is nullptr");
  auto data = value->cast<std::shared_ptr<SrcImm>>();
  MS_CHECK_TRUE_MSG(data != nullptr, RET_ERROR, "data is nullptr");
  auto data_value = data->value();
  decltype(dst_imm.value()) dst_data = static_cast<decltype(dst_imm.value())>(data_value);
  if (memcpy_s(data_info->data_.data(), sizeof(dst_imm.value()), &dst_data, sizeof(dst_imm.value())) != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed";
    return RET_MEMORY_FAILED;
  }
  return RET_OK;
}

template <typename ImmType>
int FetchImmValue(const ValueNodePtr &value_node, DataInfo *data_info) {
  MS_ASSERT(value_node != nullptr && data_info != nullptr);
  auto data = value_node->value()->cast<std::shared_ptr<ImmType>>();
  MS_CHECK_TRUE_MSG(data != nullptr, RET_NULL_PTR, "cast NumberImm failed");
  auto data_value = data->value();
  data_info->data_type_ = data->type()->number_type();
  data_info->shape_ = {1};
  data_info->data_.resize(sizeof(data_value));
  MS_CHECK_TRUE_MSG(data != nullptr, RET_NULL_PTR, "cast NumberImm failed");
  if (memcpy_s(data_info->data_.data(), sizeof(data_value), &data_value, sizeof(data_value)) != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed";
    return RET_MEMORY_FAILED;
  }
  return RET_OK;
}

int FetchFromNumberValue(const ValueNodePtr &value_node, DataInfo *data_info) {
  MS_ASSERT(value_node != nullptr && data_info != nullptr);
  data_info->data_type_ = kNumberTypeInt32;
  data_info->shape_ = {1};
  data_info->data_.resize(sizeof(int));
  auto data = value_node->value()->cast<NumberPtr>();
  MS_CHECK_TRUE_MSG(data != nullptr, RET_NULL_PTR, "cast NumberPtr failed");
  int number_type = data->number_type();
  if (TypeToTypeMap.find(number_type) != TypeToTypeMap.end()) {
    number_type = TypeToTypeMap.at(number_type);
  }
  if (memcpy_s(data_info->data_.data(), sizeof(int), &number_type, sizeof(int)) != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed";
    return RET_MEMORY_FAILED;
  }
  return RET_OK;
}

int FetchFromSequenceValue(const ValueNodePtr &value_node, DataInfo *data_info) {
  MS_ASSERT(value_node != nullptr && data_info != nullptr);
  auto value = value_node->value();
  MS_CHECK_TRUE_MSG(value != nullptr, RET_ERROR, "value is nullptr");
  std::vector<int32_t> shape;
  auto value_seq = value->cast<ValueSequencePtr>();
  MS_CHECK_TRUE_MSG(value_seq != nullptr, RET_ERROR, "value_seq is nullptr");
  if (!value_seq->value().empty()) {
    if (value_seq->value().front()->type()->number_type() == kNumberTypeInt32 ||
        value_seq->value().front()->type()->number_type() == kNumberTypeInt) {
      shape = GetValue<std::vector<int>>(value);
    } else if (value_seq->value().front()->type()->number_type() == kNumberTypeInt64) {
      auto origin_value = GetValue<std::vector<int64_t>>(value);
      std::transform(origin_value.begin(), origin_value.end(), std::back_inserter(shape),
                     [](int64_t val) { return static_cast<int32_t>(val); });
    } else {
      MS_LOG(ERROR) << "Value type is ValueSequence is not integer.";
      return RET_ERROR;
    }
  }
  data_info->data_type_ = kNumberTypeInt32;
  data_info->shape_ = {static_cast<int32_t>(shape.size())};
  data_info->data_.resize(shape.size() * sizeof(int));
  if (!shape.empty() && memcpy_s(data_info->data_.data(), shape.size() * sizeof(int32_t), shape.data(),
                                 shape.size() * sizeof(int32_t)) != EOK) {
    MS_LOG(ERROR) << "memcpy_s data into schema_tensor failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace

int FetchFromDefaultParam(const ParameterPtr &param_node, const converter::FmkType &, DataInfo *data_info,
                          bool copy_data) {
  MS_ASSERT(param_node != nullptr && data_info != nullptr);
  ShapeVector shape_vector;
  TypeId data_type = kTypeUnknown;
  auto status = GetDataTypeAndShape(param_node, &data_type, &shape_vector);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "get data type and shape from param node failed.";
    return RET_ERROR;
  }
  data_info->data_type_ = data_type;
  auto tensor_info = std::dynamic_pointer_cast<tensor::Tensor>(param_node->default_param());
  size_t offset = 0;
  if (tensor_info != nullptr && !shape_vector.empty() && data_type == kObjectTypeString) {
    status = GetShapeVectorFromStringTensor(tensor_info, &shape_vector, &offset);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "get shape vector from string tensor failed.";
      return RET_ERROR;
    }
  }
  std::vector<int32_t> dims(shape_vector.begin(), shape_vector.end());
  data_info->shape_ = dims;
  if (tensor_info != nullptr && tensor_info->Size() != 0) {
    // tensor_list tensor
    if (data_type == kObjectTypeTensorType && tensor_info->Size() >= kTensorListMinSize) {
      data_info->data_.resize(tensor_info->Size() - offset);
      if (EOK != common::huge_memcpy(data_info->data_.data(), data_info->data_.size(),
                                     static_cast<uint8_t *>(tensor_info->data_c()) + offset,
                                     tensor_info->Size() - offset)) {
        MS_LOG(ERROR) << "memcpy_s failed.";
        return RET_ERROR;
      }
    }
    // common node with const data
    if (data_type != kObjectTypeTensorType) {
      if (copy_data) {
        data_info->data_.resize(tensor_info->Size() - offset);
        if (EOK != common::huge_memcpy(data_info->data_.data(), data_info->data_.size(),
                                       static_cast<uint8_t *>(tensor_info->data_c()) + offset,
                                       tensor_info->Size() - offset)) {
          MS_LOG(ERROR) << "memcpy_s failed.";
          return RET_ERROR;
        }
      } else {
        data_info->data_ptr_ = static_cast<uint8_t *>(tensor_info->data_c()) + offset;
      }
    }
  }
  if (tensor_info != nullptr) {
    data_info->compress_type_ = tensor_info->compression_type();
  }

  data_info->format_ = NHWC;
  return RET_OK;
}

int FetchDataFromParameterNode(const CNodePtr &cnode, size_t index, converter::FmkType fmk_type, DataInfo *data_info,
                               bool copy_data) {
  MS_ASSERT(cnode != nullptr && data_info != nullptr);
  auto param_node = cnode->input(index)->cast<ParameterPtr>();
  MS_CHECK_TRUE_MSG(param_node != nullptr, RET_ERROR, "input node is not parameter node.");
  if (FetchFromDefaultParam(param_node, fmk_type, data_info, copy_data) != RET_OK) {
    MS_LOG(ERROR) << "fetch information from default param failed.";
    return RET_ERROR;
  }
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_CHECK_TRUE_MSG(prim != nullptr, RET_ERROR, "GetValueNode failed");
  if (prim->GetAttr(mindspore::ops::kFormat) == nullptr && !param_node->has_default()) {
    auto func_graph = cnode->func_graph();
    MS_CHECK_TRUE_MSG(func_graph != nullptr, RET_ERROR, "The func graph is nullptr");
    auto input_format = func_graph->get_attr(kInputFormat);
    data_info->format_ = input_format != nullptr ? GetValue<int>(input_format) : static_cast<int>(Format::NHWC);
  }
  if (prim->GetAttr(mindspore::ops::kFormat) != nullptr) {
    auto value = prim->GetAttr(mindspore::ops::kFormat);
    if (value->isa<mindspore::Int64Imm>()) {
      data_info->format_ = GetValue<int64_t>(value);
    }
  }
  QuantParamHolderPtr quant_param_holder =
    prim->GetAttr("quant_params") == nullptr ? nullptr : prim->GetAttr("quant_params")->cast<QuantParamHolderPtr>();
  if (quant_param_holder != nullptr && quant_param_holder->enable_huffman_code() &&
      data_info->data_type_ == kNumberTypeInt8) {
    data_info->enable_huffman_code_ = true;
  }
  data_info->node_type_ = NodeType_ValueNode;
  return RET_OK;
}

int FetchDataFromValueNode(const CNodePtr &cnode, size_t index, converter::FmkType fmk_type, bool train_flag,
                           DataInfo *data_info, bool copy_data) {
  MS_ASSERT(cnode != nullptr && data_info != nullptr);
  auto value_node = cnode->input(index)->cast<ValueNodePtr>();
  MS_CHECK_TRUE_MSG(value_node != nullptr, RET_ERROR, "input node is not value node.");

  auto value = value_node->value();
  int ret = RET_OK;
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_CHECK_TRUE_MSG(prim != nullptr, RET_ERROR, "prim is nullptr");
  if (value->isa<tensor::Tensor>()) {
    ret = FetchFromTensorValue(value_node, fmk_type, train_flag, data_info, copy_data);
    if (index == kNumWeightIndex && prim->GetAttr(mindspore::ops::kFormat) != nullptr) {
      data_info->format_ = GetValue<int64_t>(prim->GetAttr(mindspore::ops::kFormat));
    }
  } else if (value->isa<mindspore::Int64Imm>()) {
    ret = FetchCastImmValue<mindspore::Int32Imm, mindspore::Int64Imm>(value_node, data_info);
  } else if (value->isa<mindspore::Int32Imm>()) {
    ret = FetchImmValue<mindspore::Int32Imm>(value_node, data_info);
  } else if (value->isa<mindspore::BoolImm>()) {
    ret = FetchImmValue<mindspore::BoolImm>(value_node, data_info);
  } else if (value->isa<mindspore::FP32Imm>()) {
    ret = FetchImmValue<mindspore::FP32Imm>(value_node, data_info);
  } else if (value->isa<mindspore::ValueSequence>()) {
    ret = FetchFromSequenceValue(value_node, data_info);
  } else if (value->isa<Number>()) {
    ret = FetchFromNumberValue(value_node, data_info);
  } else if (value->isa<FuncGraph>()) {
    MS_LOG(INFO) << "op name:" << value_node->fullname_with_scope() << " input is func_graph";
    return RET_NO_CHANGE;
  } else if (value->isa<Monad>()) {
    MS_LOG(INFO) << "op name:" << value_node->fullname_with_scope() << " input is Monad";
    return RET_NO_CHANGE;
  } else {
    MS_LOG(ERROR) << "Not support value type , need add support.";
    return RET_ERROR;
  }
  data_info->node_type_ = NodeType_ValueNode;
  return ret;
}

int FetchDataFromCNode(const CNodePtr &cnode, size_t index, DataInfo *data_info) {
  MS_ASSERT(cnode != nullptr && data_info != nullptr);
  auto abstract = opt::GetCNodeInputAbstract(cnode, index);
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "Abstract cnode is nullptr.";
    return RET_ERROR;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstract)) {
    MS_LOG(ERROR) << "Abstract should be anstract tensor.";
    return RET_ERROR;
  }
  auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract);
  MS_CHECK_TRUE_MSG(abstract_tensor != nullptr, RET_ERROR, "cast ptr failed");
  auto type_ptr = abstract_tensor->element()->GetTypeTrack();
  MS_CHECK_TRUE_MSG(type_ptr != nullptr, RET_ERROR, "type_ptr is nullptr");
  if (!utils::isa<abstract::ShapePtr>(abstract_tensor->BuildShape())) {
    MS_LOG(ERROR) << "Shape of Abstract should be ShapePtr.";
    return RET_ERROR;
  }
  auto shape_vector = utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape();
  std::vector<int32_t> dims(shape_vector.begin(), shape_vector.end());
  Format format{mindspore::NHWC};
  auto ret = opt::DetermineCertainVarInputFormat(cnode, index, &format);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "set format for cnode failed";
    return RET_ERROR;
  }
  data_info->format_ = format;
  data_info->data_type_ = type_ptr->type_id();
  data_info->shape_ = dims;
  data_info->node_type_ = NodeType_CNode;
  if (type_ptr->type_id() == kObjectTypeTensorType) {
    auto tensor_info = abstract_tensor->GetValueTrack();
    if (tensor_info == nullptr || !utils::isa<tensor::TensorPtr>(tensor_info)) {
      MS_LOG(ERROR) << "tensor info is invalid.";
      return RET_ERROR;
    }
    auto tensor_value = tensor_info->cast<tensor::TensorPtr>();
    MS_CHECK_TRUE_MSG(tensor_value != nullptr, RET_ERROR, "cast ptr failed");
    if (tensor_value->Size() >= kTensorListMinSize) {
      data_info->data_.resize(tensor_value->Size());
      if (memcpy_s(data_info->data_.data(), tensor_value->Size(), tensor_value->data_c(), tensor_value->Size()) !=
          EOK) {
        MS_LOG(ERROR) << "memcpy data failed.";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

int FetchConstData(const CNodePtr &cnode, size_t index, converter::FmkType fmk_type, DataInfo *data_info,
                   bool copy_data) {
  auto node_name = cnode->fullname_with_scope();
  if (index > cnode->size()) {
    MS_LOG(ERROR) << node_name << index << " > " << cnode->size();
    return RET_ERROR;
  }
  int status;
  auto input = cnode->input(index);
  if (input->isa<Parameter>()) {
    status = FetchDataFromParameterNode(cnode, index, fmk_type, data_info, copy_data);
  } else if (input->isa<ValueNode>()) {
    status = FetchDataFromValueNode(cnode, index, fmk_type, false, data_info, copy_data);
  } else {
    MS_LOG(ERROR) << node_name << " index " << index << " is not Parameter or ValueNode";
    return RET_ERROR;
  }
  if (status != RET_OK) {
    MS_LOG(ERROR) << node_name << " fetch data failed";
    return status;
  }
  return RET_OK;
}

int RemoveIfDepend(const CNodePtr &cnode) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, RET_ERROR, "cnode is nullptr");
  bool has_depend = false;
  std::vector<AnfNodePtr> inputs;
  inputs.clear();

  inputs.emplace_back(cnode->input(0));
  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    AnfNodePtr input_node = cnode->input(i);
    MS_CHECK_TRUE_MSG(input_node != nullptr, RET_NULL_PTR, "inputNode is nullptr");
    if (!input_node->isa<CNode>()) {
      inputs.emplace_back(cnode->input(i));
      continue;
    }
    if (opt::CheckPrimitiveType(input_node, prim::kPrimDepend)) {
      auto depend_node = utils::cast<CNodePtr>(input_node);
      MS_CHECK_TRUE_MSG(depend_node != nullptr, RET_NULL_PTR, "depend_node is nullptr");
      has_depend = true;
      bool mask_out = (depend_node->inputs().size() == opt::kInputSizeThree);
      for (size_t j = 1; j < depend_node->inputs().size(); ++j) {
        AnfNodePtr depend_input_node = depend_node->input(j);
        MS_CHECK_TRUE_MSG(depend_input_node != nullptr, RET_NULL_PTR, "depend_input_node is nullptr");
        inputs.emplace_back(depend_input_node);
        if (mask_out) {
          break;
        }
      }
    } else {
      inputs.emplace_back(cnode->input(i));
    }
  }
  if (has_depend) {
    cnode->set_inputs(inputs);
  }
  return RET_OK;
}

int GetFlattenInputsIfMakeTuple(const CNodePtr &cnode, std::vector<AnfNodePtr> *inputs, bool *has_make_tuple) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, RET_NULL_PTR, "Cnode is nullptr.");
  MS_CHECK_TRUE_MSG(inputs != nullptr, RET_NULL_PTR, "Inputs is nullptr.");
  MS_CHECK_TRUE_MSG(has_make_tuple != nullptr, RET_NULL_PTR, "Has make tuple is nullptr.");
  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    AnfNodePtr input_node = cnode->input(i);
    MS_CHECK_TRUE_MSG(input_node != nullptr, RET_NULL_PTR, "Input_node is nullptr");
    auto input_cnode = utils::cast<CNodePtr>(input_node);
    if (input_cnode && (opt::CheckPrimitiveType(input_cnode, prim::kPrimMakeTuple) ||
                        opt::CheckPrimitiveType(input_cnode, opt::kPrimMakeTupleV2))) {
      *has_make_tuple = true;
      GetFlattenInputsIfMakeTuple(input_cnode, inputs, has_make_tuple);
    } else {
      inputs->emplace_back(input_node);
    }
  }
  return RET_OK;
}

int RemoveIfMakeTuple(const CNodePtr &cnode) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, RET_ERROR, "cnode is nullptr");
  bool has_make_tuple = false;
  std::vector<AnfNodePtr> inputs;
  inputs.clear();

  inputs.emplace_back(cnode->input(0));
  if (GetFlattenInputsIfMakeTuple(cnode, &inputs, &has_make_tuple) != RET_OK) {
    MS_LOG(ERROR) << "Trace real input of make tuple failed, name: " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  if (has_make_tuple) {
    cnode->set_inputs(inputs);
  }
  return RET_OK;
}

int FetchOpParameterFromNode(const AnfNodePtr &node, OpParameter **op_parameter) {
  if (op_parameter == nullptr) {
    MS_LOG(ERROR) << "op_parameter is nullptr.";
    return RET_NULL_PTR;
  }
  CHECK_NULL_RETURN(GetValueNode<PrimitivePtr>(node));
  auto prim_t = lite::GetPrimitiveT(node);
  CHECK_NULL_RETURN(prim_t);
  size_t init_size = 1024;
  flatbuffers::FlatBufferBuilder fbb(init_size);
  auto prim = lite::ConvertToPrimitive(prim_t.get(), &fbb);
  if (prim == nullptr) {
    fbb.Clear();
    MS_LOG(ERROR) << "get primitive failed.";
    return RET_ERROR;
  }
  auto parameter_gen = lite::PopulateRegistry::GetInstance()->GetParameterCreator(prim->value_type(), lite::SCHEMA_CUR);
  if (parameter_gen == nullptr) {
    fbb.Clear();
    MS_LOG(ERROR) << "PopulateParameter return nullptr, type: " << schema::EnumNamePrimitiveType(prim->value_type());
    return RET_ERROR;
  }
  *op_parameter = parameter_gen(prim);
  fbb.Clear();
  if (*op_parameter == nullptr) {
    MS_LOG(ERROR) << "parameter is nullptr.";
    return RET_ERROR;
  }
  return RET_OK;
}

int FetchOpParameterFromFuncGraph(const FuncGraphPtr &func_graph, std::map<std::string, OpParameter *> *op_parameters) {
  MS_CHECK_TRUE_MSG(op_parameters != nullptr, RET_NULL_PTR, "op_parameters is nullptr.");
  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    if (opt::IsSpecialType(cnode)) {
      continue;
    }
    auto primitive = cnode->input(0);
    OpParameter *parameter = nullptr;
    auto ret = lite::FetchOpParameterFromNode(primitive, &parameter);
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << cnode->fullname_with_scope() << " FetchOpParameterFromNode failed. ";
      return ret;
    }
    CHECK_NULL_RETURN(parameter);
    parameter->thread_num_ = 1;
    op_parameters->emplace(std::pair<std::string, OpParameter *>(cnode->fullname_with_scope(), parameter));
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
