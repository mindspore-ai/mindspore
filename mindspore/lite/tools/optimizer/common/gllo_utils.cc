/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/common/gllo_utils.h"
#include <algorithm>
#include <vector>
#include <utility>
#include <unordered_map>
#include <functional>
#include <string>
#include <set>
#include <fstream>
#include "base/float16.h"
#include "ops/fusion/conv2d_fusion.h"
#include "ops/transpose.h"
#include "ops/cast.h"
#include "ops/gather.h"
#include "ops/concat.h"
#include "ops/tuple_get_item.h"
#include "tools/common/tensor_util.h"
#include "frontend/operator/ops.h"
#include "include/backend/optimizer/helper.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "nnacl/op_base.h"
#include "src/common/log_util.h"
#include "tools/converter/parser/parser_utils.h"
#include "tools/optimizer/common/helper.h"
#include "ops/op_utils.h"
#include "ops/custom.h"
#include "include/common/utils/anfalgo.h"
#include "tools/optimizer/common/format_utils.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kAnfPrimitiveIndex = 0;
constexpr auto kDeviceTypeNone = -1;
int DeduceDimConvertion(schema::Format src_format, schema::Format dst_format, std::vector<int> *const perm) {
  MS_ASSERT(perm != nullptr);
  auto src_format_str = std::string(schema::EnumNameFormat(src_format));
  auto dst_format_str = std::string(schema::EnumNameFormat(dst_format));
  if (src_format_str.empty() || dst_format_str.empty() || src_format_str.size() != dst_format_str.size()) {
    MS_LOG(ERROR) << "src_format or dst_format is error.";
    return lite::RET_ERROR;
  }
  std::replace(src_format_str.begin(), src_format_str.end(), 'K', 'N');
  std::replace(dst_format_str.begin(), dst_format_str.end(), 'K', 'N');
  perm->clear();
  std::unordered_map<char, int> dim_map;
  for (size_t i = 0; i < src_format_str.size(); ++i) {
    dim_map[src_format_str[i]] = i;
  }
  for (size_t i = 0; i < dst_format_str.size(); ++i) {
    if (dim_map.find(dst_format_str[i]) == dim_map.end()) {
      MS_LOG(ERROR) << "src_format and dst_format cannot match, please check.";
      return RET_ERROR;
    }
    perm->push_back(dim_map[dst_format_str[i]]);
  }
  return lite::RET_OK;
}

template <class T>
void TransposeDim4(const ShapeVector &input_shape, const ShapeVector &output_shape, const std::vector<int> &perm,
                   const T *const in_data, T *out_data) {
  auto num_axes = input_shape.size();
  std::vector<int64_t> strides;
  std::vector<int64_t> out_strides;
  strides.resize(num_axes);
  out_strides.resize(num_axes);
  strides[num_axes - 1] = 1LL;
  out_strides[num_axes - 1] = 1LL;
  for (size_t i = num_axes - 1; i >= 1; i--) {
    strides[i - 1] = input_shape[i] * strides[i];
    out_strides[i - 1] = output_shape[i] * out_strides[i];
  }
  const auto stride0 = strides[perm[kIndex0]];
  const auto stride1 = strides[perm[kIndex1]];
  const auto stride2 = strides[perm[kIndex2]];
  const auto stride3 = strides[perm[kIndex3]];
  const auto out_stride0 = out_strides[kIndex0];
  const auto out_stride1 = out_strides[kIndex1];
  const auto out_stride2 = out_strides[kIndex2];
  const auto output0 = output_shape[kIndex0];
  const auto output1 = output_shape[kIndex1];
  const auto output2 = output_shape[kIndex2];
  const auto output3 = output_shape[kIndex3];

  int64_t out_beg_i = 0;
  int64_t beg_i = 0;
  for (int64_t i = 0; i < output0; ++i) {
    int64_t out_beg_ij = out_beg_i;
    int64_t beg_ij = beg_i;
    for (int64_t j = 0; j < output1; ++j) {
      int64_t out_beg_ijk = out_beg_ij;
      int64_t beg_ijk = beg_ij;
      for (int64_t k = 0; k < output2; ++k) {
        for (int64_t m = 0; m < output3; ++m) {
          out_data[out_beg_ijk + m] = in_data[beg_ijk + m * stride3];
        }
        out_beg_ijk += out_stride2;
        beg_ijk += stride2;
      }
      out_beg_ij += out_stride1;
      beg_ij += stride1;
    }
    out_beg_i += out_stride0;
    beg_i += stride0;
  }
}

template <typename T>
STATUS DoTransposeData(const tensor::TensorPtr &tensor, schema::Format src_format, schema::Format dst_format) {
  MS_ASSERT(tensor != nullptr);
  auto origin_shape = tensor->shape_c();
  if (origin_shape.size() != kInputSizeFour) {
    MS_LOG(ERROR) << "Filter dim-num is not supported, dim-num: " << origin_shape.size();
    return lite::RET_ERROR;
  }
  if (std::any_of(origin_shape.begin(), origin_shape.end(), [](int64_t val) { return val <= 0; })) {
    MS_LOG(ERROR) << "the tensor's shape is invalid.";
    return lite::RET_ERROR;
  }
  std::vector<int> perm;
  if (DeduceDimConvertion(src_format, dst_format, &perm) != RET_OK) {
    MS_LOG(ERROR) << "deduce perm failed.";
    return lite::RET_ERROR;
  }
  ShapeVector new_shape;
  for (auto &val : perm) {
    if (val < 0 || static_cast<size_t>(val) >= origin_shape.size()) {
      MS_LOG(ERROR) << "deduce perm is invalid.";
      return lite::RET_ERROR;
    }
    new_shape.push_back(origin_shape[val]);
  }
  int64_t count = 1;
  for (const auto &dat : origin_shape) {
    if (INT_MUL_OVERFLOW(count, dat)) {
      MS_LOG(ERROR) << "Int mul overflow";
      return RET_ERROR;
    }
    count *= dat;
  }
  if (count <= 0 || count > static_cast<int64_t>(INT32_MAX)) {
    MS_LOG(ERROR) << "tensor element num is too big, which should be smaller than int32_max.";
    return RET_ERROR;
  }
  std::vector<T> buf(count);

  void *originWeightData = tensor->data_c();
  MS_CHECK_TRUE_RET(originWeightData != nullptr, RET_ERROR);
  T *weightData = static_cast<T *>(originWeightData);
  TransposeDim4<T>(origin_shape, new_shape, perm, weightData, buf.data());
  if (memcpy_s(tensor->data_c(), tensor->Size(), buf.data(), count * sizeof(T)) != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed.";
    return RET_ERROR;
  }
  tensor->set_shape(new_shape);
  return RET_OK;
}

bool IsRealKernel(const AnfNodePtr &node) {
  if (node == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
  }
  // parameter and value node is not a real kernel too
  if (!node->isa<CNode>()) {
    return true;
  }
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
  }
  if (cnode->inputs().empty()) {
    MS_LOG(ERROR) << "Illegal null input of cnode(%s)" << node->DebugString();
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_INPUT_TENSOR_ERROR);
    return false;
  }
  auto input = cnode->inputs()[0];
#ifndef ENABLE_SECURITY
  bool is_virtual_node = IsPrimitive(input, prim::kPrimImageSummary) || IsPrimitive(input, prim::kPrimScalarSummary) ||
                         IsPrimitive(input, prim::kPrimTensorSummary) ||
                         IsPrimitive(input, prim::kPrimHistogramSummary) || IsPrimitive(input, prim::kPrimMakeTuple) ||
                         IsPrimitive(input, prim::kPrimStateSetItem) || IsPrimitive(input, prim::kPrimDepend) ||
                         IsPrimitive(input, prim::kPrimTupleGetItem) || IsPrimitive(input, prim::kPrimReturn) ||
                         IsPrimitive(input, prim::kPrimPartial);
#else
  bool is_virtual_node = IsPrimitive(input, prim::kPrimMakeTuple) || IsPrimitive(input, prim::kPrimStateSetItem) ||
                         IsPrimitive(input, prim::kPrimDepend) || IsPrimitive(input, prim::kPrimTupleGetItem) ||
                         IsPrimitive(input, prim::kPrimReturn) || IsPrimitive(input, prim::kPrimPartial);
#endif
  return !is_virtual_node;
}

int CopyTensorDataFromTensorInfo(const tensor::TensorPtr &tensor_info,
                                 const std::shared_ptr<tensor::Tensor> &tensor_info_dst, size_t data_count) {
  if (tensor_info->data_type() == kNumberTypeInt64) {
    auto *tensor_data = reinterpret_cast<int *>(tensor_info_dst->data_c());
    if (tensor_data == nullptr) {
      MS_LOG(ERROR) << "new data failed";
      return RET_ERROR;
    }
    auto *origin_data = reinterpret_cast<int64_t *>(tensor_info->data_c());
    MS_CHECK_TRUE_MSG(origin_data != nullptr, lite::RET_NULL_PTR, "origin_data is nullptr");
    for (size_t i = 0; i < data_count; ++i) {
      if (origin_data[i] == INT64_MAX) {
        tensor_data[i] = INT32_MAX;
      } else if (origin_data[i] == INT64_MIN) {
        tensor_data[i] = INT32_MIN;
      } else if (origin_data[i] > static_cast<int64_t>(INT32_MAX) || origin_data[i] < static_cast<int64_t>(INT32_MIN)) {
        MS_LOG(WARNING) << "int64 data " << origin_data[i] << " cannot fit into int32";
        tensor_data[i] = origin_data[i] > 0 ? INT32_MAX : INT32_MIN;
      } else {
        tensor_data[i] = static_cast<int>(origin_data[i]);
      }
    }
  } else if (tensor_info->data_type() == kNumberTypeFloat64) {
    auto *tensor_data = reinterpret_cast<float *>(tensor_info_dst->data_c());
    if (tensor_data == nullptr) {
      MS_LOG(ERROR) << "new data failed";
      return RET_ERROR;
    }
    auto *origin_data = reinterpret_cast<double_t *>(tensor_info->data_c());
    for (size_t i = 0; i < data_count; ++i) {
      if (origin_data[i] > static_cast<double_t>(FLT_MAX) || origin_data[i] < static_cast<double_t>(-FLT_MAX)) {
        MS_LOG(WARNING) << "float64 data " << origin_data[i] << " cannot fit into float32";
        tensor_data[i] = origin_data[i] > 0 ? FLT_MAX : -FLT_MAX;
      } else {
        tensor_data[i] = static_cast<float>(origin_data[i]);
      }
    }
  } else {
    tensor_info_dst->set_data_type(tensor_info->data_type());
    auto *tensor_data = reinterpret_cast<int8_t *>(tensor_info_dst->data_c());
    if (tensor_data == nullptr) {
      MS_LOG(ERROR) << "new data failed";
      return RET_ERROR;
    }
    if (memcpy_s(tensor_data, tensor_info_dst->Size(), tensor_info->data_c(), tensor_info->Size()) != lite::RET_OK) {
      MS_LOG(ERROR) << "memcpy data failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace

bool CheckInputs(const CNodePtr &cnode) {
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "cnode is nullptr.";
    return false;
  }
  if (std::any_of(cnode->inputs().begin(), cnode->inputs().end(),
                  [](const AnfNodePtr &anf_node) { return anf_node == nullptr; })) {
    MS_LOG(ERROR) << "input is nullptr.";
    return false;
  }
  return true;
}

std::vector<int> CastToInt(const ValuePtr &value) {
  if (value == nullptr) {
    MS_LOG(WARNING) << "valueptr is nullptr.";
    return {};
  }
  std::vector<int> cur_value = {};
  if (utils::isa<ValueSequencePtr>(value)) {
    if (!value->cast<ValueSequencePtr>()->value().empty()) {
      auto data_type = value->cast<ValueSequencePtr>()->value().front()->type()->number_type();
      if (data_type == kNumberTypeInt64) {
        auto origin_value = GetValue<std::vector<int64_t>>(value);
        std::transform(origin_value.begin(), origin_value.end(), std::back_inserter(cur_value),
                       [](int64_t index) { return static_cast<int>(index); });
      } else if (data_type == kNumberTypeInt || data_type == kNumberTypeInt32) {
        cur_value = GetValue<std::vector<int>>(value);
      } else {
        MS_LOG(ERROR) << "he function only process integer data.";
        return {};
      }
    }
  } else {
    auto data_type = value->type()->number_type();
    if (data_type == kNumberTypeInt64) {
      cur_value.push_back(static_cast<int>(GetValue<int64_t>(value)));
    } else if (data_type == kNumberTypeInt || data_type == kNumberTypeInt32) {
      cur_value.push_back(GetValue<int>(value));
    } else {
      MS_LOG(ERROR) << "the function only process integer data.";
      return {};
    }
  }
  return cur_value;
}

std::vector<std::vector<int>> CastToVec2DInt(const ValuePtr &value) {
  if (value == nullptr) {
    MS_LOG(WARNING) << "valueptr is nullptr.";
    return {};
  }

  std::vector<std::vector<int>> result_value;
  if (utils::isa<ValueSequencePtr>(value)) {
    auto data_type = value->cast<ValueSequencePtr>()
                       ->value()
                       .front()
                       ->cast<ValueSequencePtr>()
                       ->value()
                       .front()
                       ->type()
                       ->number_type();
    if (data_type == kNumberTypeInt64) {
      auto origin_value = GetValue<std::vector<std::vector<int64_t>>>(value);
      for (auto &i : origin_value) {
        std::vector<int> cur_value;
        std::transform(i.begin(), i.end(), std::back_inserter(cur_value),
                       [](int64_t j) { return static_cast<int>(j); });
        result_value.push_back(cur_value);
      }
    } else if (data_type == kNumberTypeInt || data_type == kNumberTypeInt32) {
      result_value = GetValue<std::vector<std::vector<int>>>(value);
    } else {
      MS_LOG(ERROR) << "he function only process integer data.";
      return result_value;
    }
  }
  return result_value;
}

std::vector<float> CastToFloat(const ValuePtr &value) {
  if (value == nullptr) {
    MS_LOG(WARNING) << "valueptr is nullptr.";
    return {};
  }
  std::vector<float> cur_value = {};
  if (utils::isa<ValueSequencePtr>(value)) {
    if (!value->cast<ValueSequencePtr>()->value().empty()) {
      auto data_type = value->cast<ValueSequencePtr>()->value().front()->type()->number_type();
      if (data_type == kNumberTypeFloat || data_type == kNumberTypeFloat32) {
        cur_value = GetValue<std::vector<float>>(value);
      } else {
        MS_LOG(ERROR) << "the function only process float data.";
        return {};
      }
    }
  } else {
    auto data_type = value->type()->number_type();
    if (data_type == kNumberTypeFloat || data_type == kNumberTypeFloat32) {
      cur_value.push_back(GetValue<float>(value));
    } else {
      MS_LOG(ERROR) << "the function only process float data.";
      return {};
    }
  }
  return cur_value;
}

bool CheckPrimitiveType(const AnfNodePtr &node, const PrimitivePtr &primitive_type) {
  if (node == nullptr || primitive_type == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
  }
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    return IsPrimitive(cnode->input(kAnfPrimitiveIndex), primitive_type);
  } else if (node->isa<ValueNode>()) {
    return IsPrimitive(node, primitive_type);
  }
  return false;
}

STATUS GetPrimitiveType(const AnfNodePtr &node, std::string *name) {
  if (name == nullptr) {
    MS_LOG(ERROR) << "name is nulltr.";
    return RET_ERROR;
  }
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (primitive == nullptr) {
      MS_LOG(ERROR) << "primitive is nullptr. " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    if (CheckPrimitiveType(node, prim::kPrimCustom)) {
      auto custom_prim = api::MakeShared<ops::Custom>(primitive);
      MS_CHECK_TRUE_MSG(custom_prim != nullptr, RET_ERROR, "custom op is nullptr.");
      *name = custom_prim->get_type();
      return RET_OK;
    } else {
      *name = primitive->name();
      return RET_OK;
    }
  } else if (node->isa<ValueNode>()) {
    auto fn_value = GetValueNode<PrimitivePtr>(node);
    CHECK_NULL_RETURN(fn_value);
    *name = fn_value->name();
    return RET_OK;
  }
  MS_LOG(ERROR) << "There is no name for this node";
  return RET_ERROR;
}

bool IsOpType(const BaseRef &n, const PrimitivePtr &prim) {
  if (utils::isa<AnfNodePtr>(n)) {
    auto anf_node = utils::cast<AnfNodePtr>(n);
    return CheckPrimitiveType(anf_node, prim);
  }
  return false;
}

bool IsRealCNodeKernel(const AnfNodePtr &node) {
  if (node == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    MS_LOG(ERROR) << "node is nullptr";
    return false;
  }
  // parameter and value node is not a real cnode kernel
  if (!node->isa<CNode>()) {
    return false;
  }
  // return considered as a real node
  if (CheckPrimitiveType(node, prim::kPrimReturn)) {
    return true;
  }
  return IsRealKernel(node);
}
bool IsGraphKernel(const AnfNodePtr &node) {
  if (node == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
  }
  // graph kernel should be a real cnode kernel.
  if (!IsRealCNodeKernel(node)) {
    return false;
  }

  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    MS_LOG(ERROR) << "node is nullptr";
    return false;
  }
  auto input = cnode->input(kAnfPrimitiveIndex);
  // graph kernel should has func_graph as first input.
  if (!IsValueNode<FuncGraph>(input)) {
    return false;
  }

  auto func_graph = GetValueNode<FuncGraphPtr>(input);
  if (func_graph == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
  }
  return func_graph->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL);
}

ParameterPtr AddNewBiasNode(const float *bias_data, const FuncGraphPtr &func_graph, int kernel_num, TypeId type_id) {
  if (bias_data == nullptr || func_graph == nullptr) {
    MS_LOG(ERROR) << "input parameter is nullptr.";
    return nullptr;
  }
  auto bias_parameter = func_graph->add_parameter();
  MS_ASSERT(bias_parameter != nullptr);
  std::vector<int64_t> shape_vector = {kernel_num};
  auto tensor_info =
    lite::CreateTensorInfo(bias_data, kernel_num * sizeof(float) / sizeof(uint8_t), shape_vector, type_id);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "create tensor info failed.";
    return nullptr;
  }
  auto status = lite::InitParameterFromTensorInfo(bias_parameter, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return nullptr;
  }

  return bias_parameter;
}

tensor::TensorPtr GetTensorInfo(const AnfNodePtr &node) {
  MS_CHECK_TRUE_RET(node != nullptr, nullptr);
  if (!utils::isa<ParameterPtr>(node)) {
    if (utils::isa<ValueNodePtr>(node)) {
      auto valueNode = node->cast<ValueNodePtr>();
      auto value_ptr = valueNode->value();
      MS_CHECK_TRUE_RET(value_ptr != nullptr, nullptr);
      auto value = value_ptr->cast<tensor::TensorPtr>();
      if (value != nullptr) {
        return value;
      }
    }
    MS_LOG(DEBUG) << "get lite param value node neither parameternode or valuenode";
    return nullptr;
  }
  auto param = node->cast<ParameterPtr>();
  MS_ASSERT(param != nullptr);
  if (!param->has_default() || param->default_param() == nullptr) {
    return nullptr;
  }
  auto tensor_info = param->default_param()->cast<tensor::TensorPtr>();
  return tensor_info;
}

AbstractBasePtr GetCNodeInputAbstract(const CNodePtr &cnode, size_t index) {
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "CNodePtr is nullptr";
    return nullptr;
  }
  auto inputs = cnode->inputs();
  if (!(index > 0 && index < inputs.size())) {
    return nullptr;
  }
  auto input = inputs[index];
  if (input == nullptr) {
    MS_LOG(ERROR) << "CNode input is nullptr";
    return nullptr;
  }

  AbstractBasePtr abstract = nullptr;
  if (utils::isa<ParameterPtr>(input)) {
    auto parameter = input->cast<ParameterPtr>();
    abstract = parameter->abstract();
  } else if (utils::isa<ValueNodePtr>(input)) {
    auto value_node = input->cast<ValueNodePtr>();
    abstract = value_node->abstract();
  } else if (utils::isa<CNodePtr>(input)) {
    auto input_cnode = input->cast<CNodePtr>();
    if (CheckPrimitiveType(input_cnode, prim::kPrimTupleGetItem)) {
      auto tuple_inputs = input_cnode->inputs();
      MS_ASSERT(tuple_inputs.size() == kTupleGetItemInputSize);
      auto get_item_input_cnode = tuple_inputs.at(1);
      MS_ASSERT(get_item_input_cnode != nullptr);
      auto idx = GetTupleGetItemOutIndex(input_cnode);
      if (!utils::isa<abstract::AbstractTuplePtr>(get_item_input_cnode->abstract())) {
        MS_LOG(ERROR) << "TupleGetItem's abstract is not AbstractTuple";
        return nullptr;
      }
      auto abstract_tuple = utils::cast<abstract::AbstractTuplePtr>(get_item_input_cnode->abstract());
      auto abstract_list = abstract_tuple->elements();
      if (abstract_list.size() <= idx) {
        MS_LOG(ERROR) << "AbstractTuple's size is smaller than expect";
        return nullptr;
      }
      abstract = abstract_list[idx];
    } else {
      abstract = input_cnode->abstract();
    }
  } else {
    MS_LOG(ERROR) << "unsupported input node type";
    return nullptr;
  }
  return abstract;
}

bool IsParamNode(const BaseRef &n) {
  if (!utils::isa<ParameterPtr>(n)) {
    return false;
  }
  auto parameter = utils::cast<ParameterPtr>(n);
  if (!parameter->has_default() || parameter->default_param() == nullptr) {
    return false;
  }
  auto tensor = parameter->default_param()->cast<tensor::TensorPtr>();
  if (tensor == nullptr) {
    return false;
  }
  return tensor->data_c() != nullptr;
}

STATUS GetTensorInfoFromAbstract(tensor::TensorPtr *const tensor_info, const CNodePtr &cnode, size_t index) {
  CHECK_NULL_RETURN(tensor_info);
  CHECK_NULL_RETURN(cnode);
  AbstractBasePtr abstract = GetCNodeInputAbstract(cnode, index);
  if (abstract == nullptr) {
    MS_LOG(WARNING) << "Abstract of CNode: " << cnode->fullname_with_scope() << " is nullptr, infershape is delayed.";
    return RET_ERROR;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstract)) {
    MS_LOG(DEBUG) << "Abstract of parameter should be abstract tensor";
    return RET_ERROR;
  }
  auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract);
  if (!utils::isa<tensor::TensorPtr>(abstract_tensor->GetValueTrack())) {  // input node not complete infershape
    MS_LOG(DEBUG) << "Value of abstract is not tensor::Tensor, indicate that infershape has failed";
    return RET_ERROR;
  }
  *tensor_info = utils::cast<tensor::TensorPtr>(abstract_tensor->GetValueTrack());
  if (*tensor_info == nullptr) {
    MS_LOG(ERROR) << "tensor::Tensor of abstract is nullptr";
    return RET_ERROR;
  }
  return RET_OK;
}

bool IsParamOrValueNodeWithData(const BaseRef &n) {
  if (utils::isa<ValueNode>(n)) {
    auto value_node = utils::cast<ValueNodePtr>(n);
    auto value = value_node->value();
    if (value == nullptr) {
      return false;
    }
    if (value->isa<tensor::Tensor>()) {
      auto tensor = value->cast<tensor::TensorPtr>();
      return tensor != nullptr && tensor->data_c() != nullptr;
    } else if (value->isa<ValueSequence>()) {
      auto sequence_ptr = value->cast<ValueSequencePtr>();
      return sequence_ptr != nullptr && !sequence_ptr->value().empty();
    } else {
      return false;
    }
  }
  if (utils::isa<ParameterPtr>(n)) {
    return IsParamNode(n);
  }
  return false;
}

bool IsParallelSplitConvNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    auto anf_node = utils::cast<AnfNodePtr>(n);
    PrimitivePtr prim;
    if (utils::isa<CNodePtr>(anf_node)) {
      prim = GetValueNode<PrimitivePtr>(anf_node->cast<CNodePtr>()->input(kAnfPrimitiveIndex));
    }
    if (utils::isa<ValueNodePtr>(anf_node)) {
      prim = GetValueNode<PrimitivePtr>(anf_node);
    }
    if (prim == nullptr) {
      MS_LOG(ERROR) << "prim is nullptr";
      return false;
    }
    int device_type =
      prim->GetAttr(ops::kDeviceType) != nullptr ? GetValue<int32_t>(prim->GetAttr(ops::kDeviceType)) : kDeviceTypeNone;
    if (device_type != kDeviceTypeNone) {
      return false;
    }
    return CheckPrimitiveType(anf_node, prim::kPrimConv2DFusion) || CheckPrimitiveType(anf_node, prim::kPrimConv2D);
  }
  return false;
}

bool IsConvNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    auto anf_node = utils::cast<AnfNodePtr>(n);
    PrimitivePtr prim;
    if (utils::isa<CNodePtr>(anf_node)) {
      prim = GetValueNode<PrimitivePtr>(anf_node->cast<CNodePtr>()->input(kAnfPrimitiveIndex));
    }
    if (utils::isa<ValueNodePtr>(anf_node)) {
      prim = GetValueNode<PrimitivePtr>(anf_node);
    }
    if (prim == nullptr) {
      MS_LOG(ERROR) << "prim is nullptr";
      return false;
    }

    if (prim->GetAttr(ops::kActivationType) != nullptr &&
        GetValue<int64_t>(prim->GetAttr(ops::kActivationType)) != NO_ACTIVATION) {
      return false;
    }

    bool is_depth_wise =
      prim->GetAttr(ops::kIsDepthWise) != nullptr && GetValue<bool>(prim->GetAttr(ops::kIsDepthWise));
    return CheckPrimitiveType(anf_node, prim::kPrimConv2DFusion) ||
           (CheckPrimitiveType(anf_node, prim::kPrimConv2dTransposeFusion) && !is_depth_wise);
  }
  return false;
}

bool CheckIsAllInputsParam(const AnfNodePtr &node) {
  if (node == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    MS_LOG(ERROR) << "node is nullptr";
    return false;
  }
  if (utils::isa<CNode>(node)) {
    auto cnode = node->cast<CNodePtr>();
    for (size_t i = 1; i < cnode->inputs().size(); i++) {
      if (!utils::isa<Parameter>(cnode->input(i)) && !utils::isa<ValueNodePtr>(cnode->input(i))) {
        return false;
      }
    }
    return true;
  }
  return false;
}

size_t GetOutputTensorNum(const AnfNodePtr &node) {
  if (node == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    MS_LOG(ERROR) << "node is nullptr";
    return 0;
  }
  auto type = node->Type();
  if (type == nullptr) {
    return 1;
  }
  if (type->isa<Tuple>()) {
    auto tuple_type = type->cast<TuplePtr>();
    if (tuple_type == nullptr) {
      lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
      MS_LOG(ERROR) << "typle_type is nullptr";
      return 0;
    }
    return tuple_type->size();
  } else if (type->isa<TensorType>() || type->isa<Number>()) {
    return 1;
  } else if (type->isa<TypeNone>()) {
    return 0;
  } else {
    return 1;
  }
}

bool IsMultiOutputTensors(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  if (graph == nullptr || node == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
  }
  auto output_node_list = Helper::GetRealNodeUsedList(graph, node);
  if (output_node_list == nullptr) {
    MS_LOG(ERROR) << "output node list is nullptr";
    return false;
  }
  if (output_node_list->size() != 1) {
    MS_LOG(DEBUG) << "fusion node has multi output nodes";
    return true;
  }
  return false;
}

AnfNodePtr GetTupleGetItemRealInput(const CNodePtr &tuple_get_item) {
  if (tuple_get_item == nullptr || tuple_get_item->size() != kInputSizeThree) {
    MS_LOG(ERROR) << "The node tuple_get_item must have 2 inputs!";
    return nullptr;
  }
  return tuple_get_item->input(kRealInputNodeIndexInTupleGetItem);
}

size_t GetTupleGetItemOutIndex(const CNodePtr &tuple_get_item) {
  if (tuple_get_item == nullptr || tuple_get_item->size() != kInputSizeThree) {
    MS_LOG(ERROR) << "The node tuple_get_item is invalid.";
    return -1;
  }
  auto output_index_value_node = tuple_get_item->input(kInputIndexTwo);
  if (output_index_value_node == nullptr) {
    MS_LOG(ERROR) << "The node tuple_get_item is invalid.";
    return -1;
  }
  auto value_node = output_index_value_node->cast<ValueNodePtr>();
  if (value_node == nullptr) {
    MS_LOG(ERROR) << "The node tuple_get_item is invalid.";
    return -1;
  }
  auto indexes = CastToInt(value_node->value());
  if (indexes.empty()) {
    MS_LOG(ERROR) << "The node tuple_get_item is invalid.";
    return -1;
  }
  return indexes.front();
}

STATUS TransFilterFormat(const tensor::TensorPtr &tensor, schema::Format src_format, schema::Format dst_format) {
  MS_CHECK_TRUE_RET(tensor != nullptr, RET_ERROR);
  std::unordered_map<TypeId, std::function<STATUS(const tensor::TensorPtr &, schema::Format, schema::Format)>>
    trans_func = {{kNumberTypeFloat32, DoTransposeData<float>},
                  {kNumberTypeUInt8, DoTransposeData<uint8_t>},
                  {kNumberTypeInt8, DoTransposeData<int8_t>},
                  {kNumberTypeFloat16, DoTransposeData<float16>}};
  auto data_type = tensor->data_type();
  auto iter = trans_func.find(data_type);
  if (iter == trans_func.end()) {
    MS_LOG(ERROR) << "Unsupported data_type: " << data_type;
    return RET_ERROR;
  }
  return iter->second(tensor, src_format, dst_format);
}

ParameterPtr BuildParameterNode(const FuncGraphPtr &func_graph, const tensor::TensorPtr &tensor_info,
                                const std::string &node_name) {
  if (func_graph == nullptr || tensor_info == nullptr) {
    MS_LOG(ERROR) << "input parameter is nullptr.";
    return nullptr;
  }
  auto param_node = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(param_node != nullptr, nullptr);
  auto shape = tensor_info->shape();
  std::vector<int64_t> shape_vector;
  std::transform(shape.begin(), shape.end(), std::back_inserter(shape_vector),
                 [](const int &val) { return static_cast<int64_t>(val); });
  auto data_type = tensor_info->data_type();
  if (tensor_info->data_type() == kNumberTypeInt64) {
    data_type = kNumberTypeInt32;
  } else if (tensor_info->data_type() == kNumberTypeFloat64) {
    data_type = kNumberTypeFloat32;
  }
  param_node->set_name(node_name);
  auto tensor_info_new = std::make_shared<tensor::Tensor>(data_type, shape_vector);
  if (tensor_info_new == nullptr) {
    MS_LOG(ERROR) << "new tensor::Tensor failed.";
    return nullptr;
  }
  int data_count = 1;
  for (const auto &dat : shape) {
    if (INT_MUL_OVERFLOW(data_count, static_cast<int>(dat))) {
      MS_LOG(ERROR) << "Int mul overflow.";
      return nullptr;
    }
    data_count *= static_cast<int>(dat);
  }
  if (data_count < 0) {
    MS_LOG(ERROR) << "Invalid shape.";
    return nullptr;
  }
  if (tensor_info->Size() == 0) {
    auto status = lite::InitParameterFromTensorInfo(param_node, tensor_info_new);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "init parameter from tensor info failed";
      return nullptr;
    }
    return param_node;
  }

  if (CopyTensorDataFromTensorInfo(tensor_info, tensor_info_new, static_cast<size_t>(data_count)) != RET_OK) {
    MS_LOG(ERROR) << "copy tensor data failed";
    return nullptr;
  }

  auto status = lite::InitParameterFromTensorInfo(param_node, tensor_info_new);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return nullptr;
  }
  param_node->set_default_param(tensor_info_new);
  return param_node;
}

ParameterPtr BuildIntValueParameterNode(const FuncGraphPtr &func_graph, const int32_t &data,
                                        const std::string &node_name, bool empty_shape) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  auto param_node = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(param_node != nullptr, nullptr);
  param_node->set_name(node_name);
  ShapeVector shape = empty_shape ? std::vector<int64_t>{} : std::vector<int64_t>{1};
  auto tensor_info = lite::CreateTensorInfo(&data, sizeof(int32_t), shape, kNumberTypeInt32);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Create tensor info failed";
    return nullptr;
  }

  auto status = lite::InitParameterFromTensorInfo(param_node, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return nullptr;
  }
  return param_node;
}

ParameterPtr BuildIntVecParameterNode(const FuncGraphPtr &func_graph, const std::vector<int32_t> &data,
                                      const std::string &node_name) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  auto param_node = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(param_node != nullptr, nullptr);
  param_node->set_name(node_name);

  std::vector<int64_t> shape_vector{static_cast<int64_t>(data.size())};
  auto tensor_info = lite::CreateTensorInfo(data.data(), data.size() * sizeof(int32_t), shape_vector, kNumberTypeInt32);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Create tensor info failed";
    return nullptr;
  }

  auto status = lite::InitParameterFromTensorInfo(param_node, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return nullptr;
  }

  return param_node;
}

ParameterPtr BuildIntVec2DParameterNode(const FuncGraphPtr &func_graph, const std::vector<std::vector<int32_t>> &data,
                                        const std::string &node_name) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  auto param_node = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(param_node != nullptr, nullptr);
  param_node->set_name(node_name);

  MS_CHECK_TRUE_RET(!data.empty(), nullptr);
  std::vector<int64_t> shape_vector;
  shape_vector.push_back(data.size());
  shape_vector.push_back(data.at(0).size());

  std::vector<int32_t> data_1d;
  for (auto pair : data) {
    data_1d.insert(data_1d.end(), pair.begin(), pair.end());
  }

  auto size = data_1d.size() * sizeof(int32_t);
  auto tensor_info = lite::CreateTensorInfo(data_1d.data(), size, shape_vector, kNumberTypeInt32);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Create tensor info failed";
    return nullptr;
  }
  auto status = lite::InitParameterFromTensorInfo(param_node, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return nullptr;
  }
  return param_node;
}

ParameterPtr BuildFloatValueParameterNode(const FuncGraphPtr &func_graph, const float &data,
                                          const std::string &node_name, bool empty_shape) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  auto param_node = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(param_node != nullptr, nullptr);
  param_node->set_name(node_name);

  ShapeVector shape = empty_shape ? std::vector<int64_t>{} : std::vector<int64_t>{1};
  auto tensor_info = lite::CreateTensorInfo(&data, sizeof(float), shape, kNumberTypeFloat32);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Create tensor info failed";
    return nullptr;
  }
  auto status = lite::InitParameterFromTensorInfo(param_node, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return nullptr;
  }
  return param_node;
}

ParameterPtr BuildFloatVecParameterNode(const FuncGraphPtr &func_graph, const std::vector<float> &data,
                                        const std::string &node_name) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  auto param_node = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(param_node != nullptr, nullptr);
  param_node->set_name(node_name);

  std::vector<int64_t> shape_vector{static_cast<int64_t>(data.size())};
  auto tensor_info = lite::CreateTensorInfo(data.data(), data.size() * sizeof(float), shape_vector, kNumberTypeFloat);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Create tensor info failed";
    return nullptr;
  }

  auto status = lite::InitParameterFromTensorInfo(param_node, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return nullptr;
  }

  return param_node;
}

CNodePtr GenTransposeNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node, const std::vector<int> &perm,
                          const std::string &cnode_name) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(input_node != nullptr, nullptr);
  auto perm_node = BuildIntVecParameterNode(func_graph, perm, cnode_name + "_perm");
  MS_ASSERT(perm_node != nullptr);
  ops::Transpose transpose_node;
  auto trans_prim = transpose_node.GetPrim();
  MS_CHECK_TRUE_RET(trans_prim != nullptr, nullptr);
  auto cnode = func_graph->NewCNode(trans_prim, {input_node, perm_node});
  MS_ASSERT(cnode != nullptr);
  auto manager = Manage(func_graph);
  MS_ASSERT(manager != nullptr);
  manager->SetEdge(cnode, 1, input_node);
  manager->SetEdge(cnode, kInputIndexTwo, perm_node);
  cnode->set_fullname_with_scope(cnode_name);
  auto quant_params_holder = std::make_shared<lite::QuantParamHolder>(kInputSizeTwo, 1);
  MS_CHECK_TRUE_RET(quant_params_holder != nullptr, nullptr);
  trans_prim->AddAttr("quant_params", quant_params_holder);
  return cnode;
}

CNodePtr GenCastNode(const FuncGraphPtr &graph, const AnfNodePtr &input_node, const std::string &cnode_name,
                     const TypeId dst_type, const AbstractBasePtr &abstract) {
  MS_CHECK_TRUE_RET(graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(input_node != nullptr, nullptr);
  // auto new_cast = std::make_shared<mindspore::ops::Cast>();
  ops::Cast cast_node;
  auto new_cast_c = cast_node.GetPrim();
  if (new_cast_c == nullptr) {
    MS_LOG(ERROR) << "new_cast_c is nullptr";
    return nullptr;
  }
  ValueNodePtr value_node = NewValueNode(new_cast_c);
  if (value_node == nullptr) {
    MS_LOG(ERROR) << "NewValueNode Failed";
    return nullptr;
  }

  auto param_node = opt::BuildIntValueParameterNode(graph, static_cast<int32_t>(dst_type), cnode_name + "_type");

  auto cast_cnode = graph->NewCNode({value_node});
  if (cast_cnode == nullptr) {
    MS_LOG(ERROR) << "new_cnode is nullptr";
    return nullptr;
  }
  cast_cnode->set_fullname_with_scope(cnode_name);
  cast_cnode->set_abstract(abstract);
  auto manager = Manage(graph);
  (void)manager->Replace(input_node, cast_cnode);
  manager->AddEdge(cast_cnode, input_node);
  manager->AddEdge(cast_cnode, param_node);
  return cast_cnode;
}

CNodePtr GenGatherNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node, const std::vector<int> &indices,
                       const std::string &cnode_name) {
  if (func_graph == nullptr || input_node == nullptr) {
    MS_LOG(ERROR) << "input parameter is nullptr, which is invalid.";
    return nullptr;
  }
  auto indices_node = BuildIntVecParameterNode(func_graph, indices, cnode_name + "_indices");
  if (indices_node == nullptr) {
    MS_LOG(ERROR) << "make indices node failed.";
    return nullptr;
  }
  auto axis_node = BuildIntVecParameterNode(func_graph, {0}, cnode_name + "_axis");
  if (axis_node == nullptr) {
    MS_LOG(ERROR) << "make indices node failed.";
    return nullptr;
  }
  ops::Gather gather_node;
  auto gather_prim = gather_node.GetPrim();
  MS_CHECK_TRUE_RET(gather_prim != nullptr, nullptr);
  auto cnode = func_graph->NewCNode(gather_prim, {input_node, indices_node, axis_node});
  MS_ASSERT(cnode != nullptr);
  auto manager = Manage(func_graph);
  MS_ASSERT(manager != nullptr);
  manager->SetEdge(cnode, 1, input_node);
  manager->SetEdge(cnode, kInputIndexTwo, indices_node);
  manager->SetEdge(cnode, kInputIndexThree, axis_node);
  cnode->set_fullname_with_scope(cnode_name);
  auto quant_params_holder = std::make_shared<lite::QuantParamHolder>(kInputSizeThree, 1);
  MS_CHECK_TRUE_RET(quant_params_holder != nullptr, nullptr);
  gather_prim->AddAttr("quant_params", quant_params_holder);
  return cnode;
}

CNodePtr GenConcatNode(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &input_node_vec,
                       const std::string &cnode_name) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "func_graph is nullptr, which is invalid.";
    return nullptr;
  }
  ops::Concat concat_node;
  concat_node.set_axis(0);
  auto concat_prim = concat_node.GetPrim();
  MS_CHECK_TRUE_RET(concat_prim != nullptr, nullptr);
  auto cnode = func_graph->NewCNode(concat_prim, input_node_vec);
  MS_ASSERT(cnode != nullptr);
  auto manager = Manage(func_graph);
  MS_ASSERT(manager != nullptr);
  cnode->set_fullname_with_scope(cnode_name);
  auto quant_params_holder = std::make_shared<lite::QuantParamHolder>(input_node_vec.size(), 1);
  MS_CHECK_TRUE_RET(quant_params_holder != nullptr, nullptr);
  concat_prim->AddAttr("quant_params", quant_params_holder);
  return cnode;
}

CNodePtr GenTupleGetItemNode(const FuncGraphPtr &func_graph, const CNodePtr &input, size_t index) {
  if (func_graph == nullptr || input == nullptr) {
    MS_LOG(ERROR) << "input parameter is nullptr, which is invalid.";
    return nullptr;
  }
  auto tuple_get_item_prim = std::make_shared<ops::TupleGetItem>();
  MS_CHECK_TRUE_RET(tuple_get_item_prim != nullptr, nullptr);
  auto second_input = NewValueNode(MakeValue<int>(index));
  MS_CHECK_TRUE_RET(second_input != nullptr, nullptr);
  auto tuple_get_item_prim_c = tuple_get_item_prim->GetPrim();
  MS_CHECK_TRUE_RET(tuple_get_item_prim_c != nullptr, nullptr);
  auto tuple_cnode = func_graph->NewCNode(tuple_get_item_prim_c, {input, second_input});
  MS_ASSERT(tuple_cnode != nullptr);
  tuple_cnode->set_fullname_with_scope(input->fullname_with_scope() + "_getitem_" + std::to_string(index));
  return tuple_cnode;
}

STATUS FetchShapeFromAbstract(const abstract::AbstractBasePtr &abstract, ShapeVector *shape) {
  if (abstract == nullptr || shape == nullptr) {
    MS_LOG(ERROR) << "input parameter is nullptr, which is invalid.";
    return lite::RET_ERROR;
  }
  if (!utils::isa<abstract::AbstractTensor>(abstract)) {
    MS_LOG(ERROR) << "abstract of cnode is invalid.";
    return lite::RET_ERROR;
  }
  auto abstract_tensor = abstract->cast<abstract::AbstractTensorPtr>();
  if (abstract_tensor->BuildShape() == nullptr || !utils::isa<abstract::ShapePtr>(abstract_tensor->BuildShape())) {
    MS_LOG(ERROR) << "shape of cnode's output is invalid.";
    return lite::RET_ERROR;
  }
  *shape = utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape();
  return lite::RET_OK;
}

bool IsTrainOp(const CNodePtr &cnode) {
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (prim == nullptr) {
    return false;
  }
  auto cnode_type = prim->name();
  // optimizer op
  if (cnode_type == "Adam" || cnode_type == "SGD" || cnode_type == "ApplyMomentum") {
    return true;
  }
  // loss op
  if (cnode_type == "SoftmaxCrossEntropyWithLogits" || cnode_type == "SparseSoftmaxCrossEntropyWithLogits" ||
      cnode_type == "SmoothL1Loss" || cnode_type == "SmoothL1LossGrad" ||
      cnode_type == "SigmoidCrossEntropyWithLogits" || cnode_type == "SigmoidCrossEntropyWithLogitsGrad") {
    return true;
  }
  // grad op
  if (cnode_type.find("Grad") != std::string::npos ||
      cnode->fullname_with_scope().find("Gradients") != std::string::npos) {
    return true;
  }
  return false;
}

bool IsMarkedTrainOp(const CNodePtr &cnode) {
  if (cnode == nullptr) {
    return false;
  }
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_CHECK_TRUE_RET(prim != nullptr, false);
  if (prim->GetAttr("trainOp") != nullptr && GetValue<bool>(prim->GetAttr("trainOp"))) {
    MS_LOG(DEBUG) << "train op not fusion.";
    return true;
  }
  return false;
}

size_t GetOutputSize(const AnfNodePtr &anf_node) {
  if (anf_node == nullptr) {
    MS_LOG(ERROR) << "anf_node is nullptr.";
    return RET_ERROR;
  }
  AbstractBasePtr abstract_base;
  if (CheckPrimitiveType(anf_node, prim::kPrimTupleGetItem)) {
    abstract_base = anf_node->cast<CNodePtr>()->input(1)->abstract();
  } else {
    abstract_base = anf_node->abstract();
  }
  // used for multi output e.g. split.
  if (utils::isa<abstract::AbstractTuple>(abstract_base)) {
    auto abstract_tuple = abstract_base->cast<abstract::AbstractTuplePtr>();
    return abstract_tuple->elements().size();
  }
  return 1;
}

ShapeVector GetAnfNodeOutputShape(const AnfNodePtr &node, size_t output_idx) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "anf_node is nullptr.";
    return {};
  }
  auto as_value_node = node->cast<ValueNodePtr>();
  if (as_value_node) {
    auto value = as_value_node->value();
    auto tensor = value->cast<tensor::TensorPtr>();
    if (tensor) {
      return tensor->shape_c();
    }
    return {};
  }
  auto base_shape = node->Shape();
  if (base_shape == nullptr) {
    MS_LOG(INFO) << "Failed to get shape from node " << node->fullname_with_scope();
    return {};
  }
  if (base_shape->isa<abstract::Shape>()) {
    if (output_idx != 0) {
      MS_LOG(EXCEPTION) << "The node " << node->fullname_with_scope() << "is a single output node but got index ["
                        << output_idx;
    }
    auto shape_ptr = base_shape->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape_ptr);
    return shape_ptr->shape();
  } else if (base_shape->isa<abstract::NoShape>()) {
    return ShapeVector();
  } else if (base_shape->isa<abstract::TupleShape>()) {
    auto tuple_shape = base_shape->cast<abstract::TupleShapePtr>();
    MS_EXCEPTION_IF_NULL(tuple_shape);
    if (output_idx >= tuple_shape->size()) {
      MS_LOG(EXCEPTION) << "Output index " << output_idx << "is larger than output number " << tuple_shape->size()
                        << node->fullname_with_scope();
    }
    auto b_shp = (*tuple_shape)[output_idx];
    if (b_shp->isa<abstract::Shape>()) {
      auto shape_ptr = b_shp->cast<abstract::ShapePtr>();
      MS_EXCEPTION_IF_NULL(shape_ptr);
      return shape_ptr->shape();
    } else if (b_shp->isa<abstract::NoShape>()) {
      return ShapeVector();
    } else if (b_shp->isa<abstract::TupleShape>()) {
      MS_LOG(INFO) << "The output shape of node:" << node->fullname_with_scope() << " index:" << output_idx
                   << " is a TupleShape:" << base_shape->ToString();
      return ShapeVector();
    } else {
      MS_LOG(EXCEPTION) << "The output type of ApplyKernel index:" << output_idx
                        << " should be a NoShape , ArrayShape or a TupleShape, but it is " << base_shape->ToString()
                        << "node :" << node->fullname_with_scope() << ".";
    }
  }
  return ShapeVector();
}

int GetDataTypeFromAnfNode(const AnfNodePtr &anf_node, TypeId *type_id) {
  if (anf_node == nullptr || type_id == nullptr) {
    MS_LOG(ERROR) << "anf_node or type_id is nullptr.";
    return RET_ERROR;
  }
  AbstractBasePtr abstract_base;
  if (CheckPrimitiveType(anf_node, prim::kPrimTupleGetItem)) {
    abstract_base = anf_node->cast<CNodePtr>()->input(1)->abstract();
  } else {
    abstract_base = anf_node->abstract();
  }
  // used for multi output e.g. split.
  if (utils::isa<abstract::AbstractTuple>(abstract_base)) {
    auto abstract_tuple = abstract_base->cast<abstract::AbstractTuplePtr>();
    if (abstract_tuple->elements().empty()) {
      MS_LOG(ERROR) << "abstract_tuple elements is empty.";
      return RET_ERROR;
    }
    abstract_base = abstract_tuple->elements().front();
  }
  if (abstract_base == nullptr) {
    MS_LOG(INFO) << "Abstract of parameter is nullptr, " << anf_node->fullname_with_scope();
    *type_id = kTypeUnknown;
    return lite::RET_NOT_SUPPORT;
  }
  if (utils::isa<abstract::AbstractTensorPtr>(abstract_base)) {
    auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract_base);
    MS_CHECK_TRUE_MSG(abstract_tensor != nullptr, RET_ERROR, "Cast to abstract tensor failed!");
    auto type_ptr = abstract_tensor->element()->GetTypeTrack();
    MS_CHECK_TRUE_MSG(type_ptr != nullptr, RET_ERROR, "type_ptr is nullptr");
    *type_id = type_ptr->type_id();
  } else if (utils::isa<abstract::AbstractScalarPtr>(abstract_base)) {
    auto abstract_scalar = utils::cast<abstract::AbstractScalarPtr>(abstract_base);
    auto type_ptr = abstract_scalar->GetTypeTrack();
    MS_CHECK_TRUE_MSG(type_ptr != nullptr, RET_ERROR, "type_ptr is nullptr");
    *type_id = type_ptr->type_id();
  } else {
    MS_LOG(ERROR) << anf_node->fullname_with_scope() << " is unsupported type:" << abstract_base->type_name();
    return RET_ERROR;
  }
  return RET_OK;
}

bool IsQuantParameterNode(const PrimitivePtr &prim) {
  MS_CHECK_TRUE_RET(prim != nullptr, false);
  auto quant_attr = prim->GetAttr("quant_params");
  if (quant_attr != nullptr) {
    auto quant_param_holder = quant_attr->cast<lite::QuantParamHolderPtr>();
    MS_CHECK_TRUE_RET(quant_param_holder != nullptr, false);
    auto quant_params = quant_param_holder->get_input_quant_params();
    bool is_quant = std::any_of(quant_params.begin(), quant_params.end(), [](std::vector<schema::QuantParamT> &params) {
      return !params.empty() && params.front().inited;
    });
    if (is_quant) {
      return true;
    }
  }
  return false;
}

void UpdateManager(const FuncGraphPtr &func_graph) {
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    manager = Manage(func_graph, true);
  } else {
    manager->Clear();
    manager->AddFuncGraph(func_graph, true);
  }
  std::set<FuncGraphPtr> all_func_graphs;
  mindspore::lite::GetAllFuncGraph(func_graph, &all_func_graphs);
  for (auto &one_func_graph : all_func_graphs) {
    manager->AddFuncGraph(one_func_graph);
  }
}

std::pair<CNodePtr, int> GetRealCertainVarInput(const CNodePtr &cnode, size_t index) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, {}, "function's parameter is nullptr.");
  MS_CHECK_TRUE_MSG(cnode->input(index) != nullptr, {}, "required input is nullptr");
  auto real_input_cnode = cnode->input(index)->cast<CNodePtr>();
  MS_CHECK_TRUE_MSG(real_input_cnode != nullptr, {}, "input node is not a cnode.");
  int item_index = 0;
  if (opt::CheckPrimitiveType(real_input_cnode, prim::kPrimTupleGetItem)) {
    auto index_node = real_input_cnode->input(opt::kInputIndexTwo);
    MS_CHECK_TRUE_MSG(index_node != nullptr, {}, "tuple_get_item's second input is nullptr.");
    MS_CHECK_TRUE_MSG(index_node->isa<ValueNode>(), {}, "tuple_get_item's second input should be valuenode.");
    auto index_ptr = index_node->cast<ValueNodePtr>()->value();
    MS_CHECK_TRUE_MSG(index_ptr != nullptr, {}, "tuple_get_item's second input val is nullptr.");
    auto value = CastToInt(index_ptr);
    MS_CHECK_TRUE_MSG(value.size() == 1, {}, "tuple_get_item's second input is invalid.");
    item_index = value.front();
    MS_CHECK_TRUE_MSG(real_input_cnode->input(1) != nullptr, {}, "tuple_get_item's first input is nullptr");
    real_input_cnode = real_input_cnode->input(1)->cast<CNodePtr>();
    MS_CHECK_TRUE_MSG(real_input_cnode != nullptr, {}, "tuple_get_item first input is not cnode.");
  }
  return {real_input_cnode, item_index};
}

int DetermineCertainVarInputHasInferred(const CNodePtr &cnode, size_t index, bool *infer_succ) {
  MS_CHECK_TRUE_MSG(cnode != nullptr && infer_succ != nullptr, RET_ERROR, "function's parameter is nullptr.");
  auto var_input_info = GetRealCertainVarInput(cnode, index);
  if (var_input_info.first == nullptr) {
    MS_LOG(ERROR) << "cannot get the real var input.";
    return RET_ERROR;
  }
  auto real_input_cnode = var_input_info.first;
  auto item_index = var_input_info.second;
  auto input_node_prim = GetValueNode<PrimitivePtr>((real_input_cnode->input(0)));
  MS_CHECK_TRUE_MSG(input_node_prim != nullptr, RET_ERROR, "get primitive failed.");
  *infer_succ = false;
  auto value_ptr = input_node_prim->GetAttr(kInferDone);
  if (value_ptr != nullptr) {
    MS_CHECK_TRUE_MSG(value_ptr->isa<BoolImm>(), RET_ERROR, "value is not a boolean.");
    *infer_succ = GetValue<bool>(value_ptr);
  }
  value_ptr = input_node_prim->GetAttr(kInferFlags);
  if (value_ptr == nullptr) {
    return RET_OK;
  }
  MS_CHECK_TRUE_MSG(value_ptr->isa<ValueSequeue>(), RET_ERROR, "infer flag should be a vector.");
  auto value_sequence = value_ptr->cast<ValueSequeuePtr>();
  auto elements = value_sequence->value();
  MS_CHECK_TRUE_MSG(!elements.empty(), RET_ERROR, "infer_info has no content.");
  auto first_element = elements.front();
  MS_CHECK_TRUE_MSG(first_element != nullptr, RET_ERROR, "element is a nullptr.");
  MS_CHECK_TRUE_MSG(first_element->isa<BoolImm>(), RET_ERROR, "each element is not a boolean.");
  auto infer_infos = GetValue<std::vector<bool>>(value_ptr);
  MS_CHECK_TRUE_MSG(item_index >= 0 && static_cast<size_t>(item_index) < infer_infos.size(), RET_ERROR,
                    "item index is out of range.");
  *infer_succ = infer_infos[item_index];
  return RET_OK;
}
bool CheckAndGetCnodeIndex(const CNodePtr &cnode, size_t *index, const PrimitivePtr &primitive_type) {
  MS_CHECK_TRUE_RET(cnode != nullptr, false);
  MS_CHECK_TRUE_RET(index != nullptr, false);
  if (cnode->size() != kInputSizeThree) {
    return false;
  }
  size_t dst_index = 0;
  for (size_t i = 1; i < cnode->size(); ++i) {
    if (CheckPrimitiveType(cnode->input(i), primitive_type)) {
      dst_index = i;
      break;
    }
  }
  if (dst_index == 0) {
    return false;
  }
  *index = dst_index;
  return true;
}

void PrintFuncGraph(const FuncGraphPtr &func_graph, const std::string &output_file) {
  if (func_graph == nullptr) {
    MS_LOG_WARNING << "input func_graph is nullptr";
    return;
  }
  static int index = 0;
  auto real_file = std::to_string(index++) + "_" + output_file + ".txt";
  std::ofstream fp(real_file);
  if (!fp.is_open()) {
    MS_LOG(ERROR) << "Failed to create file " << real_file;
    return;
  }
  auto nodes = func_graph->TopoSort(func_graph->get_return());
  auto type_name = [](const AnfNodePtr &anf_node) -> std::string {
    if (anf_node->cast<CNodePtr>()) {
      return GetCNodeFuncName(anf_node->cast<CNodePtr>());
    } else if (anf_node->cast<ParameterPtr>()) {
      if (anf_node->cast<ParameterPtr>()->has_default()) {
        return "Parameter_Constant";
      } else {
        return "Parameter_Variable";
      }
    } else if (anf_node->cast<ValueNodePtr>()) {
      return "ValueNode";
    }
    return anf_node->ToString();
  };
  for (auto &node : nodes) {
    if (IsValueNode<Primitive>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      fp << node->fullname_with_scope() << ", type: " << type_name(node)
         << ", shape: " << GetAnfNodeOutputShape(node, 0) << std::endl;
      fp << std::endl;
      continue;
    }
    TypeId type_id = kTypeUnknown;
    GetDataTypeFromAnfNode(node, &type_id);
    fp << node->fullname_with_scope() << ", type: " << type_name(node) << ", shape: " << GetAnfNodeOutputShape(node, 0)
       << ", data type: " << static_cast<int>(type_id) << std::endl;
    auto inputs = cnode->inputs();
    for (auto &input : inputs) {
      if (IsValueNode<Primitive>(input)) {
        continue;
      }
      type_id = kTypeUnknown;
      GetDataTypeFromAnfNode(node, &type_id);
      fp << "---input " << input->fullname_with_scope() << ", type: " << type_name(input)
         << ", shape: " << GetAnfNodeOutputShape(input, 0) << ", data type: " << static_cast<int>(type_id) << std::endl;
    }
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (prim != nullptr) {
      for (auto &attr : prim->attrs()) {
        if (attr.second) {
          fp << "---attr " << attr.first << ": " << attr.second->ToString() << std::endl;
        } else {
          fp << "---attr " << attr.first << ": value nullptr" << std::endl;
        }
      }
    }
    fp << std::endl;
  }
}

#if !defined(_WIN32) && !defined(_WIN64)
std::vector<KernelWithIndex> GetNodeInputs(const AnfNodePtr &anf_node) {
  if (!anf_node) {
    return {};
  }
  if (!anf_node->isa<CNode>()) {
    return {{anf_node, 0}};
  }
  auto cnode = anf_node->cast<CNodePtr>();
  std::vector<common::KernelWithIndex> inputs;
  size_t input_num = common::AnfAlgo::GetInputTensorNum(cnode);
  for (size_t input_idx = 0; input_idx < input_num; ++input_idx) {
    const auto &pre_node_output = common::AnfAlgo::GetPrevNodeOutput(cnode, input_idx);
    auto pre_node = pre_node_output.first;
    if (opt::CheckPrimitiveType(pre_node, prim::kPrimMakeTuple) ||
        opt::CheckPrimitiveType(pre_node, opt::kPrimMakeTupleV2)) {
      auto tuple_inputs = GetNodeInputs(pre_node);
      std::copy(tuple_inputs.begin(), tuple_inputs.end(), std::back_inserter(inputs));
    } else {
      inputs.push_back(pre_node_output);
    }
  }
  return inputs;
}
#endif

bool IsReduceModeMeetOutEqualIn(const PrimitivePtr &prim) {
  if (prim == nullptr) {
    return false;
  }
  if (prim->GetAttr(ops::kMode) == nullptr) {
    return false;
  }
  auto mode = GetValue<int64_t>(prim->GetAttr(ops::kMode));
  std::set<int64_t> meet_mode = {Reduce_Mean, Reduce_Max, Reduce_Min, Reduce_Prod, Reduce_Sum};
  return meet_mode.find(mode) != meet_mode.end();
}
}  // namespace opt
}  // namespace mindspore
