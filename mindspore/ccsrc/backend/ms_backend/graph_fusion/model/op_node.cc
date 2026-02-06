/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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
#include "backend/ms_backend/graph_fusion/model/op_node.h"

#include <cmath>
#include <algorithm>
#include <memory>
#include <set>
#include <sstream>
#include <functional>
#include <numeric>
#include <utility>
#include <string>
#include <vector>

#include "ir/tensor_new.h"
#include "ir/dtype/tensor_type.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/anf_utils.h"
#include "utils/hash_map.h"
#include "utils/check_convert_utils.h"
#include "backend/ms_backend/graph_fusion/core/graph_kernel_utils.h"
#include "backend/ms_backend/graph_fusion/model/node.h"
#include "utils/log_adapter.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"

namespace mindspore::graphkernel::inner {
std::vector<int64_t> GetListInt(const ValuePtr &attr_value) {
  std::vector<int64_t> list_int;
  const auto &val_seq = attr_value->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(val_seq);
  const auto &vals = val_seq->value();
  (void)std::transform(vals.begin(), vals.end(), std::back_inserter(list_int),
                       [](const ValuePtr &v) { return AnfUtils::GetIntValue(v); });
  return list_int;
}

BaseShapePtr InferShapeWithAbstract(const PrimitivePtr &prim, const AbstractBasePtrList &abs_list) {
  auto shape_optional = abstract::InferShapeByFuncImpl(prim, abs_list, true);
  if (shape_optional.has_value()) {
    return shape_optional.value();
  }

  auto found = abstract::GetPrimitiveInferImpl(prim);
  if (found.has_value()) {
    auto infer = found.value();
    if (infer.IsImplInferShapeAndType()) {
      return infer.InferShape(prim, abs_list);
    }
  }
  MS_LOG(EXCEPTION) << "The infer function of [" << prim->name() << "] is not defined.";
  return nullptr;
}

TypePtr InferTypeWithAbstract(const PrimitivePtr &prim, const AbstractBasePtrList &abs_list) {
  auto type_optional = abstract::InferTypeByFuncImpl(prim, abs_list, true);
  if (type_optional.has_value()) {
    return type_optional.value();
  }

  auto found = abstract::GetPrimitiveInferImpl(prim);
  if (found.has_value()) {
    auto infer = found.value();
    if (infer.IsImplInferShapeAndType()) {
      return infer.InferType(prim, abs_list);
    }
  }
  MS_LOG(EXCEPTION) << "The infer function of [" << prim->name() << "] is not defined.";
  return nullptr;
}

tensor::TensorPtr InferValueWithAbstract(const PrimitivePtr &prim, const AbstractBasePtrList &abs_list) {
  auto value_optional = abstract::InferValueByFuncImpl(prim, abs_list);
  if (value_optional.has_value()) {
    return std::static_pointer_cast<tensor::Tensor>(value_optional.value());
  }

  auto found = abstract::GetPrimitiveInferImpl(prim);
  if (found.has_value()) {
    auto infer = found.value();
    if (infer.IsImplInferValue()) {
      return std::static_pointer_cast<tensor::Tensor>(infer.InferValue(prim, abs_list));
    }
  }
  return nullptr;
}

std::pair<PrimitivePtr, AbstractBasePtrList> PrimOp::GenPrimAndAbstract(const NodePtrList &inputs,
                                                                        const DAttrs &attrs) const {
  auto prim = std::make_shared<Primitive>(op_);
  MS_EXCEPTION_IF_NULL(prim);
  (void)prim->SetAttrs(attrs);
  AbstractBasePtrList abs_list(inputs.size());
  (void)std::transform(inputs.cbegin(), inputs.cend(), abs_list.begin(),
                       [](const NodePtr &node) { return node->ToAbstract(); });
  return std::make_pair(prim, abs_list);
}

std::vector<DShape> PrimOp::InferShape(const NodePtrList &inputs, const DAttrs &attrs) {
  auto [prim, abs_list] = GenPrimAndAbstract(inputs, attrs);
  RectifyAbstract(prim, &abs_list);
  auto baseshape = InferShapeWithAbstract(prim, abs_list);
  MS_EXCEPTION_IF_NULL(baseshape);
  if (baseshape->isa<abstract::TupleShape>()) {
    auto tuple_shape = baseshape->cast<abstract::TupleShapePtr>();
    MS_EXCEPTION_IF_NULL(tuple_shape);
    const auto &shape_elements = tuple_shape->shape();
    std::vector<DShape> result(shape_elements.size());
    (void)std::transform(shape_elements.cbegin(), shape_elements.cend(), result.begin(),
                         [](const BaseShapePtr &s) { return s->cast<abstract::ShapePtr>()->shape(); });
    return result;
  }
  auto shape = baseshape->cast<abstract::ShapePtr>();
  if (shape != nullptr) {
    return {shape->shape()};
  }
  return {DShape()};
}

std::vector<TypeId> PrimOp::InferType(const NodePtrList &inputs, const DAttrs &attrs) {
  auto [prim, abs_list] = GenPrimAndAbstract(inputs, attrs);
  RectifyAbstract(prim, &abs_list);
  auto type = InferTypeWithAbstract(prim, abs_list);
  MS_EXCEPTION_IF_NULL(type);
  auto get_type_id = [](const TypePtr &t) {
    return t->isa<TensorType>() ? t->cast<TensorTypePtr>()->element()->type_id() : t->type_id();
  };
  if (type->isa<Tuple>()) {
    auto elements = type->cast<TuplePtr>()->elements();
    std::vector<TypeId> result(elements.size());
    (void)std::transform(elements.cbegin(), elements.cend(), result.begin(), get_type_id);
    return result;
  }
  return {get_type_id(type)};
}

NodeBaseList PrimOp::Infer(const NodePtrList &inputs, const DAttrs &attrs) {
  Check(inputs, attrs);
  NodeBaseList result;
  auto format = InferFormat(inputs, attrs);
  auto shapes = InferShape(inputs, attrs);
  auto types = InferType(inputs, attrs);
  if (shapes.size() != types.size()) {
    MS_LOG(EXCEPTION) << "The num of shapes and types should be equal. (" << shapes.size() << " vs " << types.size()
                      << ")";
  }
  for (size_t i = 0; i < shapes.size(); i++) {
    (void)result.emplace_back(NodeBase{shapes[i], types[i], format});
  }
  return result;
}

std::string PrimOp::ToString() const {
  std::ostringstream oss;
  oss << Node::ToString();
  oss << " = " << this->op_ << "(";
  for (size_t i = 0; i < inputs_.size(); i++) {
    if (inputs_[i]->NodeType() == NType::Primitive) {
      oss << inputs_[i]->Node::ToString();
    } else {
      oss << inputs_[i]->ToString();
    }
    if (i != inputs_.size() - 1) {
      oss << ", ";
    }
  }
  oss << ")";
  std::ostringstream attr_oss;
  bool has_attr = false;
  std::set<std::string> black_list = {"IsFeatureMapInputList", "IsFeatureMapOutput", "output_names", "input_names"};
  for (auto attr : attrs_) {
    if (attr.second != nullptr && black_list.count(attr.first) == 0) {
      if (has_attr) {
        attr_oss << ", ";
      } else {
        has_attr = true;
      }
      attr_oss << attr.first << ": " << attr.second->ToString();
    }
  }
  if (has_attr) {
    oss << "  // attr {" << attr_oss.str() << "}";
  }
  return oss.str();
}

template <typename TD, typename TE>
std::vector<TE> ChangeDataToVec(const NodePtr &n) {
  std::vector<TE> res;
  TD *data = static_cast<TD *>(std::static_pointer_cast<inner::ConstTensorNode>(n)->data()->data_c());
  for (size_t elem = 0; elem < n->tensor_size(); elem++) {
    res.push_back(static_cast<TE>(*(data + elem)));
  }
  return res;
}

template <typename TM>
tensor::TensorPtr PrimOp::CalcByOperator(const NodePtrList &inputs, const DAttrs &) const {
  const size_t unary_input_num = 1;
  const size_t binary_input_num = 2;
  if (inputs.size() > 0) {
    bool all_shape_equal =
      std::all_of(inputs.begin(), inputs.end(), [&inputs](const NodePtr &t) { return t->shape == inputs[0]->shape; });
    if (!all_shape_equal) {
      return nullptr;
    }
  }
  std::vector<std::vector<TM>> inputs_tm;
  const auto &op = this->op();
  const auto tid = this->type;
  for (const auto &t : inputs) {
    (void)inputs_tm.emplace_back(ChangeDataToVec<TM, TM>(t));
  }
  if (inputs.size() == unary_input_num) {
    mindspore::HashMap<std::string, std::function<TM(const TM &)>> func_map = {
      {"Abs", [](const TM &a) { return a <= TM(0) ? -a : a; }},
      {"Exp", [](const TM &a) { return exp(a); }},
      {"Log", [](const TM &a) { return log(a); }},
      {"Neg", [](const TM &a) { return -a; }},
      {"Reciprocal",
       [](const TM &a) {
         if (a == TM(0)) {
           MS_LOG(EXCEPTION) << "During graph kernel constant fold for reciprocal, divisor is zero.";
         }
         return TM(1) / a;
       }},
      {"Rsqrt",
       [](const TM &a) {
         if (a == TM(0)) {
           MS_LOG(EXCEPTION) << "During graph kernel constant fold for rsqrt, divisor is zero.";
         }
         return TM(1) / sqrt(a);
       }},
      {"Sqrt", [](const TM &a) { return sqrt(a); }},
    };
    if (func_map.find(op) == func_map.end()) {
      return nullptr;
    }
    const auto &input_a = inputs_tm[0];
    std::vector<TM> res;
    (void)std::transform(input_a.begin(), input_a.end(), std::back_inserter(res),
                         [&func_map, &op](const TM &i) { return func_map[op](i); });
    return tensor::from_buffer(tid, this->shape, &res[0], tid);
  } else if (inputs.size() == binary_input_num) {
    mindspore::HashMap<std::string, std::function<TM(const TM &, const TM &)>> func_map = {
      {"Add", [](const TM &a, const TM &b) { return a + b; }},
      {"Sub", [](const TM &a, const TM &b) { return a - b; }},
      {"Mul", [](const TM &a, const TM &b) { return a * b; }},
      {"RealDiv",
       [](const TM &a, const TM &b) {
         if (b == TM(0)) {
           MS_LOG(EXCEPTION) << "During graph kernel constant fold for realdiv, divisor is zero.";
         }
         return a / b;
       }},
    };
    if (func_map.find(op) == func_map.end()) {
      return nullptr;
    }
    const auto &input_a = inputs_tm[0];
    const auto &input_b = inputs_tm[1];
    std::vector<TM> res;
    for (size_t i = 0; i < input_a.size(); i++) {
      (void)res.emplace_back(func_map[op](input_a[i], input_b[i]));
    }
    return tensor::from_buffer(tid, this->shape, &res[0], tid);
  }
  return nullptr;
}

NodePtr PrimOp::InferValue(const NodePtrList &inputs, const DAttrs &attrs) {
  for (auto i : inputs) {
    if (i->NodeType() != NType::Tensor) {
      return nullptr;
    }
  }
  TypeId output_type = this->type;
  tensor::TensorPtr res = nullptr;
  switch (static_cast<int>(output_type)) {
    case TypeId::kNumberTypeUInt8: {
      res = CalcByOperator<uint8_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeInt8: {
      res = CalcByOperator<int8_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeInt16: {
      res = CalcByOperator<int16_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeInt32: {
      res = CalcByOperator<int32_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeInt64: {
      res = CalcByOperator<int64_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeUInt16: {
      res = CalcByOperator<uint16_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeUInt32: {
      res = CalcByOperator<uint32_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeUInt64: {
      res = CalcByOperator<uint64_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeFloat16: {
      res = CalcByOperator<float16>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeFloat32: {
      res = CalcByOperator<float>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeFloat64: {
      res = CalcByOperator<double>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeBFloat16: {
      res = CalcByOperator<bfloat16>(inputs, attrs);
      break;
    }
    default:
      return nullptr;
  }
  if (res == nullptr) {
    auto [prim, inputs_abstract] = GenPrimAndAbstract(inputs, attrs);
    RectifyAbstract(prim, &inputs_abstract);
    res = InferValueWithAbstract(prim, inputs_abstract);
  }
  return res == nullptr ? nullptr : std::make_shared<ConstTensorNode>(res);
}

NodePtr ReshapeOp::InferValue(const NodePtrList &inputs, const DAttrs &) {
  if (inputs[0]->NodeType() != NType::Tensor) {
    return nullptr;
  }
  void *tensor_data = inputs[0]->As<inner::ConstTensorNode>()->data()->data_c();
  tensor::TensorPtr result_tensor = tensor::from_buffer(this->type, this->shape, tensor_data, this->type);
  return std::make_shared<ConstTensorNode>(result_tensor);
}

// default format shape to fractal_Nz format shape
DShape ToNz(const DShape &default_shape) {
  constexpr size_t nz_size = 2;
  constexpr auto align16 = 16;
  auto len = default_shape.size();
  DShape leading_shape;
  DShape tail_shape;
  if (default_shape.size() == 1 && default_shape[0] == 1) {
    // # As shape (1,) can broadcast to any shape, it can be regarded as a special FractalNZ shape
    return default_shape;
  }
  if (default_shape.size() > nz_size) {
    (void)leading_shape.insert(leading_shape.cend(), default_shape.cbegin(),
                               default_shape.cend() - SizeToLong(nz_size));
  }
  if (default_shape.size() == 1 || (default_shape.size() >= nz_size && default_shape[len - nz_size] == 1)) {
    // (32) or (N, 1, 32) -> (N, 2, 1, 1, 16)
    if (default_shape.back() % align16 != 0) {
      MS_LOG(EXCEPTION) << "default_shape[-1] should be multiplies of 16, but got " << default_shape.back();
    }
    tail_shape = {default_shape.back() / align16, 1, 1, align16};
  } else if (default_shape.size() >= nz_size || default_shape[1] == 1) {
    // (N, 32, 1) -> (N, 1, 2, 16, 1)
    if (default_shape[len - nz_size] % align16 != 0) {
      MS_LOG(EXCEPTION) << "default_shape[-2] should be multiplies of 16, but got " << default_shape[len - nz_size];
    }
    tail_shape = {1, default_shape[0] / align16, align16, 1};
  } else {
    // (N, 32, 48) -> (N, 3, 2, 16, 16)
    if (default_shape.back() % align16 != 0 || default_shape[len - nz_size] % align16 != 0) {
      MS_LOG(EXCEPTION) << "default_shape[-1] and default_shape[-2]should be multiplies of 16, but got "
                        << default_shape.back() << " " << default_shape[len - nz_size];
    }
    tail_shape = {default_shape[1] / align16, default_shape[0] / align16, align16, align16};
  }
  (void)leading_shape.insert(leading_shape.cend(), tail_shape.cbegin(), tail_shape.cend());
  return leading_shape;
}

DShape BroadcastShape(const NodePtrList &inputs, bool to_nz = false) {
  std::vector<std::vector<int64_t>> shapes;
  for (auto &input : inputs) {
    if (to_nz && input->format != kOpFormat_FRAC_NZ) {
      (void)shapes.emplace_back(ToNz(input->shape));
    } else {
      (void)shapes.emplace_back(input->shape);
    }
  }
  auto max_dim_input =
    std::max_element(shapes.begin(), shapes.end(),
                     [](const std::vector<int64_t> &a, const std::vector<int64_t> &b) { return a.size() < b.size(); });
  auto max_dim = max_dim_input->size();
  std::vector<std::vector<int64_t>> align_shapes;
  for (auto &s : shapes) {
    std::vector<int64_t> cur(max_dim - s.size(), 1);
    (void)cur.insert(cur.cend(), s.cbegin(), s.cend());
    (void)align_shapes.emplace_back(cur);
  }
  std::vector<int64_t> output_shape(max_dim, 1);
  for (size_t i = 0; i < max_dim; i++) {
    for (auto &align_shape : align_shapes) {
      if (align_shape[i] > 1) {
        if (output_shape[i] == 1) {
          output_shape[i] = align_shape[i];
        }
        if (output_shape[i] != align_shape[i]) {
          MS_LOG(EXCEPTION) << "Shape broadcast failed: " << output_shape[i] << " vs " << align_shape[i];
        }
      }
    }
  }
  return output_shape;
}

std::vector<DShape> ElemwiseOp::InferShape(const NodePtrList &inputs, const DAttrs &attrs) {
  if (std::any_of(inputs.begin(), inputs.end(),
                  [](const NodePtr &input) { return input->format == kOpFormat_FRAC_NZ; })) {
    return {BroadcastShape(inputs, true)};
  }
  return PrimOp::InferShape(inputs, attrs);
}

DFormat ElemwiseOp::InferFormat(const NodePtrList &inputs, const DAttrs &) {
  if (inputs.empty()) {
    return kOpFormat_DEFAULT;
  }
  auto first_format = inputs[0]->format;
  for (const auto &inp : inputs) {
    auto cur_format = inp->format;
    if (cur_format.find("FRACTAL") != std::string::npos) {
      // special format
      return cur_format;
    }
    if (cur_format != kOpFormat_DEFAULT && inp->tensor_size() != 1) {
      return cur_format;
    }
  }
  return first_format;
}

std::vector<DShape> ArgReduceOp::InferShape(const NodePtrList &inputs, const DAttrs &attrs) {
  CHECK_ATTR(attrs, "axis");
  auto axis = GetListInt(attrs.find("axis")->second);
  const auto &input_shape = inputs[0]->shape;
  int64_t size = SizeToLong(input_shape.size());
  std::vector<int64_t> real_axis;
  (void)std::transform(axis.begin(), axis.end(), std::back_inserter(real_axis),
                       [&size](const int64_t &x) { return x < 0 ? (x + size) : x; });

  DShape new_shape;
  for (size_t i = 0; i < input_shape.size(); i++) {
    if (std::find(real_axis.begin(), real_axis.end(), SizeToLong(i)) == real_axis.end()) {
      (void)new_shape.emplace_back(input_shape[i]);
    }
  }
  if (new_shape.empty()) {
    (void)new_shape.emplace_back(1);
  }
  return {new_shape};
}

std::vector<TypeId> ArgReduceOp::InferType(const NodePtrList &, const DAttrs &attrs) {
  CHECK_ATTR(attrs, "output_type");
  return {attrs.find("output_type")->second->cast<TypePtr>()->type_id()};
}

DFormat TransposeOp::InferFormat(const NodePtrList &inputs, const DAttrs &attrs) {
  if (attrs.count(kAttrDstFormat) != 0) {
    return GetValue<std::string>(attrs.find(kAttrDstFormat)->second);
  }
  // only support NCHW/NHWC now
  constexpr size_t kRank4 = 4;
  if (inputs[0]->shape.size() != kRank4) {
    return kOpFormat_DEFAULT;
  }
  auto perm_node = inputs[1];
  auto perm_tensor = perm_node->As<inner::ConstTensorNode>()->data();
  auto perm = CheckAndConvertUtils::CheckTensorIntValue("permutation", perm_tensor, "Transpose");
  const auto &ori_format = inputs[0]->format;
  if (ori_format == kOpFormat_DEFAULT || ori_format == kOpFormat_NCHW) {
    std::vector<int64_t> nchw2nhwc = {0, 2, 3, 1};
    if (perm == nchw2nhwc) {
      return kOpFormat_NHWC;
    }
  } else if (ori_format == kOpFormat_NHWC) {
    std::vector<int64_t> nhwc2nchw = {0, 3, 1, 2};
    if (perm == nhwc2nchw) {
      return kOpFormat_NCHW;
    }
  }
  return kOpFormat_DEFAULT;
}

NodePtr ShapeOp::InferValue(const NodePtrList &inputs, const DAttrs &) {
  auto tensor = tensor::from_buffer(this->type, this->shape, inputs[0]->shape.data(), kNumberTypeInt64);
  return std::make_shared<ConstTensorNode>(tensor);
}

std::vector<DShape> PadAkgOp::InferShape(const NodePtrList &inputs, const DAttrs &attrs) {
  std::vector<int64_t> shape0 = inputs[0]->shape;
  size_t n = shape0.size();
  CHECK_ATTR(attrs, "head");
  CHECK_ATTR(attrs, "tail");
  std::vector<int64_t> pad_before = GetListInt(attrs.find("head")->second);
  std::vector<int64_t> pad_after = GetListInt(attrs.find("tail")->second);
  if (pad_before.size() != n || pad_after.size() != n) {
    MS_LOG(EXCEPTION) << "Input dimension and pad mismatch: " << n << " vs " << pad_before.size() << " vs "
                      << pad_after.size();
  }
  std::vector<int64_t> output;
  for (size_t i = 0; i < n; i++) {
    (void)output.emplace_back(shape0[i] + pad_before[i] + pad_after[i]);
  }
  return {output};
}

std::vector<DShape> UnPadAkgOp::InferShape(const NodePtrList &inputs, const DAttrs &attrs) {
  std::vector<int64_t> shape0 = inputs[0]->shape;
  size_t n = shape0.size();
  CHECK_ATTR(attrs, "tail");
  std::vector<int64_t> unpad_after = GetListInt(attrs.find("tail")->second);
  if (unpad_after.size() != n) {
    MS_LOG(EXCEPTION) << "Input dimension and pad mismatch: " << n << " vs " << unpad_after.size();
  }
  std::vector<int64_t> output;
  for (size_t i = 0; i < n; i++) {
    (void)output.emplace_back(shape0[i] - unpad_after[i]);
  }
  return {output};
}

bool Conv2dOp::HadPad(const ShapeVector &pad_list, const std::string &pad_mode) {
  constexpr size_t kTop = 0;
  constexpr size_t kBottom = 1;
  constexpr size_t kLeft = 2;
  constexpr size_t kRight = 3;

  if (pad_list[kTop] != pad_list[kBottom] || pad_list[kLeft] != pad_list[kRight]) {
    return true;
  }
  if (pad_mode != "VALID" && pad_mode != "valid") {
    return std::any_of(pad_list.begin(), pad_list.end(), [](auto a) { return a != 0; });
  }
  return false;
}

std::vector<DShape> Conv2dOp::InferShape(const NodePtrList &inputs, const DAttrs &attrs) {
  // get the output shape when format is NHWC/NCHW
  if (inputs[0]->shape.size() == kDim4) {
    CHECK_ATTR(attrs, "format");
    if (inputs[0]->format == kOpFormat_NHWC || inputs[1]->format == kOpFormat_NHWC ||
        GetValue<std::string>(attrs.find("format")->second) == kOpFormat_NHWC) {
      CHECK_ATTR(attrs, "pad_mode");
      CHECK_ATTR(attrs, "pad_list");
      CHECK_ATTR(attrs, "kernel_size");
      CHECK_ATTR(attrs, "stride");
      CHECK_ATTR(attrs, "dilation");

      auto x_shape = inputs[0]->shape;
      auto w_shape = inputs[1]->shape;
      auto pad_mode = GetValue<std::string>(attrs.find("pad_mode")->second);
      auto pad_list = GetListInt(attrs.find("pad_list")->second);
      auto kernel_size = GetListInt(attrs.find("kernel_size")->second);
      auto stride = GetListInt(attrs.find("stride")->second);
      auto dilation = GetListInt(attrs.find("dilation")->second);
      constexpr size_t kPadSize = 4;
      constexpr size_t kKernelSize = 2;
      constexpr size_t kStrideSize = 4;
      constexpr size_t kDilationSize = 4;
      if (x_shape.size() != kDim4 || w_shape.size() != kDim4 || pad_list.size() != kPadSize ||
          kernel_size.size() != kKernelSize || stride.size() != kStrideSize || dilation.size() != kDilationSize) {
        MS_LOG(EXCEPTION) << "For 'Conv2D', got sizes of x_shape, w_shape, pad_list, kernel_size, stride and dilation: "
                          << x_shape.size() << ", " << w_shape.size() << ", " << pad_list.size() << ", "
                          << kernel_size.size() << ", " << stride.size() << ", " << dilation.size()
                          << ". But expect: 4, 4, 4, 2, 4, 4";
      }
      auto has_pad = HadPad(pad_list, pad_mode);
      if (!has_pad) {
        pad_list = {0, 0, 0, 0};
      }

      auto k_h = (kernel_size[0] - 1) * dilation[2] + 1;
      auto k_w = (kernel_size[1] - 1) * dilation[3] + 1;
      auto out_h = (x_shape[1] + pad_list[0] + pad_list[1] - k_h) / stride[2] + 1;
      auto out_w = (x_shape[2] + pad_list[2] + pad_list[3] - k_w) / stride[3] + 1;
      return {{x_shape[0], out_h, out_w, w_shape[3]}};
    } else {
      return OpaqueOp::InferShape(inputs, attrs);
    }
  }

  // get the output shape when format is NCHWc
  std::vector<int64_t> data_shape = inputs[0]->shape;
  std::vector<int64_t> weight_shape = inputs[1]->shape;
  auto n = data_shape[0];
  auto i_h = data_shape[2];
  auto i_w = data_shape[3];
  auto c_o_o = weight_shape[0];
  auto k_h = weight_shape[2];
  auto k_w = weight_shape[3];
  auto c_o_i = weight_shape[5];

  CHECK_ATTR(attrs, "stride");
  CHECK_ATTR(attrs, "dilation");

  std::vector<int64_t> strides = GetListInt(attrs.find("stride")->second);
  std::vector<int64_t> dilations = GetListInt(attrs.find("dilation")->second);

  auto d_h = dilations[0];
  auto d_w = dilations[1];
  auto s_h = strides[0];
  auto s_w = strides[1];
  auto k_h_d = (k_h - 1) * d_h + 1;
  auto k_w_d = (k_w - 1) * d_w + 1;
  auto o_h = (i_h - k_h_d) / s_h + 1;
  auto o_w = (i_w - k_w_d) / s_w + 1;

  std::vector<int64_t> output_shape{n, c_o_o, o_h, o_w, c_o_i};
  return {output_shape};
}

std::vector<TypeId> Conv2dOp::InferType(const NodePtrList &inputs, const DAttrs &attrs) {
  if (inputs[0]->shape.size() == kDim4) {
    return PrimOp::InferType(inputs, attrs);
  }
  return {inputs[0]->type};
}

DFormat Conv2dOp::InferFormat(const NodePtrList &inputs, const DAttrs &attrs) {
  if (inputs[0]->shape.size() == kDim4) {
    return PrimOp::InferFormat(inputs, attrs);
  }
  CHECK_ATTR(attrs, "conv_out_format");
  return GetValue<std::string>(attrs.find("conv_out_format")->second);
}

void ConcatOp::RectifyAbstract(const PrimitivePtr &, AbstractBasePtrList *input_abstract_ptr) {
  AbstractBasePtrList rectifyed_abs_list;
  (void)rectifyed_abs_list.emplace_back(std::make_shared<abstract::AbstractTuple>(*input_abstract_ptr));
  input_abstract_ptr->swap(rectifyed_abs_list);
}

void ReduceOp::RectifyAbstract(const PrimitivePtr &prim, AbstractBasePtrList *abs_list) {
  CHECK_ATTR(prim->attrs(), "keep_dims");
  (void)abs_list->emplace_back(prim->GetAttr("keep_dims")->ToAbstract());
  if (prim->name() == prim::kPrimReduceSum->name()) {
    CHECK_ATTR(prim->attrs(), "skip_mode");
    (void)abs_list->emplace_back(prim->GetAttr("skip_mode")->ToAbstract());
  }
}

void ArgReduceOp::RectifyAbstract(const PrimitivePtr &prim, AbstractBasePtrList *abs_list) {
  CHECK_ATTR(prim->attrs(), "axis");
  (void)abs_list->emplace_back(prim->GetAttr("axis")->ToAbstract());
  CHECK_ATTR(prim->attrs(), "output_type");
  (void)abs_list->emplace_back(prim->GetAttr("output_type")->ToAbstract());
}

std::vector<size_t> CompactShape(const ShapeVector &origin, int64_t axis) {
  std::vector<size_t> new_shape;
  size_t accu = 1;
  for (size_t i = 0; i < origin.size(); i++) {
    if (LongToSize(axis) == i) {
      new_shape.push_back(accu);
      new_shape.push_back(LongToSize(origin[i]));
      accu = 1;
    } else {
      accu *= LongToSize(origin[i]);
    }
  }
  new_shape.push_back(accu);
  return new_shape;
}

template <typename TM>
tensor::TensorPtr ConcatOp::CalcConcat(const NodePtrList &inputs, const DAttrs &attrs) {
  constexpr size_t first_dim = 0;
  constexpr size_t second_dim = 1;
  constexpr size_t third_dim = 2;
  int64_t axis = 0;
  auto axis_node = inputs.back();
  if (axis_node->NodeType() == NType::Scalar) {
    auto scalar_node = axis_node->As<ConstScalarNode>();
    axis = GetValue<int64_t>(scalar_node->data());
  } else {
    return nullptr;
  }
  axis = axis < 0 ? SizeToLong(this->shape.size()) + axis : axis;
  std::vector<std::vector<TM>> inputs_tm;
  for (const auto &t : inputs) {
    (void)inputs_tm.emplace_back(ChangeDataToVec<TM, TM>(t));
  }
  std::vector<std::vector<size_t>> all_shps;
  (void)std::transform(inputs.begin(), inputs.end(), std::back_inserter(all_shps),
                       [&axis](const NodePtr &t) { return CompactShape(t->shape, axis); });
  std::vector<TM> res;
  if (all_shps.size() > 0) {
    const size_t third_dim_size = all_shps[0][third_dim];
    const size_t first_dim_size = all_shps[0][first_dim];
    for (size_t i = 0; i < first_dim_size; i++) {
      for (size_t t = 0; t < inputs_tm.size(); t++) {
        for (size_t j = 0; j < all_shps[t][second_dim]; j++) {
          for (size_t k = 0; k < third_dim_size; k++) {
            (void)res.emplace_back(inputs_tm[t][i * all_shps[t][second_dim] * third_dim_size + j * third_dim_size + k]);
          }
        }
      }
    }
    return tensor::from_buffer(this->type, this->shape, &res[0], this->type);
  }
  return nullptr;
}

NodePtr ConcatOp::InferValue(const NodePtrList &inputs, const DAttrs &attrs) {
  for (auto i : inputs) {
    if (i->NodeType() != NType::Tensor) {
      return nullptr;
    }
  }
  TypeId output_type = this->type;
  tensor::TensorPtr res = nullptr;
  switch (static_cast<int>(output_type)) {
    case TypeId::kNumberTypeUInt8: {
      res = CalcConcat<uint8_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeInt8: {
      res = CalcConcat<int8_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeInt16: {
      res = CalcConcat<int16_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeInt32: {
      res = CalcConcat<int32_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeInt64: {
      res = CalcConcat<int64_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeUInt16: {
      res = CalcConcat<uint16_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeUInt32: {
      res = CalcConcat<uint32_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeUInt64: {
      res = CalcConcat<uint64_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeFloat16: {
      res = CalcConcat<float16>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeFloat32: {
      res = CalcConcat<float>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeFloat64: {
      res = CalcConcat<double>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeBFloat16: {
      res = CalcConcat<bfloat16>(inputs, attrs);
      break;
    }
    default:
      return nullptr;
  }
  return res == nullptr ? nullptr : std::make_shared<ConstTensorNode>(res);
}

void ComplexOp::Check(const NodePtrList &inputs, const DAttrs &) {
  if (inputs[0]->type != TypeId::kNumberTypeFloat32) {
    MS_LOG(EXCEPTION) << "Complex's input[0] should be float32, but got " << TypeIdToString(inputs[0]->type, true);
  }
  if (inputs[0]->type != inputs[1]->type) {
    MS_LOG(EXCEPTION) << "Complex's input[0] and inputs[1]'s type mismatch: " << TypeIdToString(inputs[0]->type, true)
                      << " vs " << TypeIdToString(inputs[1]->type, true);
  }
}

std::vector<DShape> StandardNormalOp::InferShape(const NodePtrList &, const DAttrs &attrs) {
  CHECK_ATTR(attrs, "shape");
  return {GetListInt(attrs.find("shape")->second)};
}

void MatMulOp::RectifyAbstract(const PrimitivePtr &prim, AbstractBasePtrList *abs_list) {
  CHECK_ATTR(prim->attrs(), "transpose_a");
  (void)abs_list->emplace_back(prim->GetAttr("transpose_a")->ToAbstract());
  CHECK_ATTR(prim->attrs(), "transpose_b");
  (void)abs_list->emplace_back(prim->GetAttr("transpose_b")->ToAbstract());
}

std::vector<DShape> MatMulOp::InferShape(const NodePtrList &inputs, const DAttrs &attrs) {
  // the prim's infer shape does not supports batch dims
  constexpr size_t kMatMulRank = 2;
  DShape rev_shape[kDim2] = {inputs[0]->shape, inputs[1]->shape};
  auto size = std::max(rev_shape[0].size(), rev_shape[1].size());
  DShape output_shape(size);
  for (size_t i = 0; i < kDim2; i++) {
    std::reverse(rev_shape[i].begin(), rev_shape[i].end());
    rev_shape[i].resize(size, 1);
  }
  for (size_t i = kMatMulRank; i < size; i++) {
    if (rev_shape[0][i] != 1 && rev_shape[1][i] != 1) {
      MS_LOG(EXCEPTION) << "The Matmul's batch dim is unmatched. got " << inputs[0]->shape << " and "
                        << inputs[1]->shape;
    }
    output_shape[i] = std::max(rev_shape[0][i], rev_shape[1][i]);
  }
  auto tranA = GetValue<bool>(attrs.at(kTransposeA));
  auto tranB = GetValue<bool>(attrs.at(kTransposeB));
  output_shape[1] = tranA ? rev_shape[0][0] : rev_shape[0][1];
  output_shape[0] = tranB ? rev_shape[1][1] : rev_shape[1][0];

  std::reverse(output_shape.begin(), output_shape.end());

  if (inputs.size() > kSizeTwo &&
      (inputs[kSizeTwo]->shape.size() != 1 || inputs[kSizeTwo]->shape[0] != output_shape.back())) {
    MS_LOG(EXCEPTION) << "The MatmulBiasAdd's shape is unmatched. got " << output_shape << " and "
                      << inputs[kSizeTwo]->shape;
  }
  return {output_shape};
}

std::vector<TypeId> MatMulOp::InferType(const NodePtrList &inputs, const DAttrs &attrs) {
  if (attrs.count("dst_type") != 0) {
    return {attrs.find("dst_type")->second->cast<TypePtr>()->type_id()};
  }
  if (inputs[0]->type == TypeId::kNumberTypeInt8) {
    return {TypeId::kNumberTypeInt32};
  }
  return {inputs[0]->type};
}
}  // namespace mindspore::graphkernel::inner
