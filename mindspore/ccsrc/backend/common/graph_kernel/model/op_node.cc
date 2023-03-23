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
#include "backend/common/graph_kernel/model/op_node.h"

#include <cmath>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <sstream>
#include <functional>
#include <numeric>
#include <utility>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/anf_utils.h"
#include "utils/hash_map.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "backend/common/graph_kernel/model/node.h"
#include "backend/operator/ops_backend_infer_function.h"

namespace mindspore::graphkernel::inner {
std::vector<int64_t> GetListInt(const ValuePtr &attr_value) {
  std::vector<int64_t> list_int;
  const auto &vals = attr_value->cast<ValueSequencePtr>()->value();
  (void)std::transform(vals.begin(), vals.end(), std::back_inserter(list_int),
                       [](const ValuePtr &v) { return AnfUtils::GetIntValue(v); });
  return list_int;
}

BaseShapePtr InferShapeWithAbstract(const PrimitivePtr &prim, const AbstractBasePtrList &abs_list) {
  auto found = abstract::GetBackendPrimitiveInferImpl(prim);
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
  auto found = abstract::GetBackendPrimitiveInferImpl(prim);
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
  auto found = abstract::GetBackendPrimitiveInferImpl(prim);
  if (found.has_value()) {
    auto infer = found.value();
    if (infer.IsImplInferValue()) {
      return std::static_pointer_cast<tensor::Tensor>(infer.InferValue(prim, abs_list));
    }
  }
  return nullptr;
}

void PrimOp::SetAbastractsFromAttrs(const PrimitivePtr &primitive, const mindspore::HashSet<size_t> &convert_input_list,
                                    AbstractBasePtrList *inputs_abstract, std::vector<std::string> input_names_vec) {
  for (size_t index = 0; index < input_names_vec.size(); ++index) {
    // if convert input list find the index it means the input has been converted to the attr
    if (convert_input_list.find(index) != convert_input_list.end()) {
      AbstractBasePtr rectify_abs = nullptr;
      auto input_name = input_names_vec[index];
      auto attr = primitive->GetAttr(input_name);
      if (attr != nullptr) {
        rectify_abs = attr->ToAbstract();
        inputs_abstract->push_back(rectify_abs);
      }
    }
  }
}

std::pair<PrimitivePtr, AbstractBasePtrList> PrimOp::GenPrimAndAbstract(const NodePtrList &inputs,
                                                                        const DAttrs &attrs) const {
  const auto &op_primc_fns = ops::OpPrimCRegister::GetInstance().GetPrimCMap();
  const auto iter = op_primc_fns.find(op_);
  if (iter == op_primc_fns.end()) {
    return std::pair<PrimitivePtr, AbstractBasePtrList>(nullptr, {});
  }
  auto primc = iter->second();
  (void)primc->SetAttrs(attrs);
  AbstractBasePtrList abs_list(inputs.size());
  (void)std::transform(inputs.cbegin(), inputs.cend(), abs_list.begin(),
                       [](const NodePtr &node) { return node->ToAbstract(); });
  return std::make_pair(primc->cast<PrimitivePtr>(), abs_list);
}

std::vector<DShape> PrimOp::InferShape(const NodePtrList &inputs, const DAttrs &attrs) {
  auto [primc, abs_list] = GenPrimAndAbstract(inputs, attrs);
  if (primc == nullptr) {
    MS_LOG(EXCEPTION) << "The PrimitiveC of [" << op_ << "] is not defined.";
  }
  RectifyAbstract(primc, &abs_list);
  auto baseshape = InferShapeWithAbstract(primc, abs_list);
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
  auto [primc, abs_list] = GenPrimAndAbstract(inputs, attrs);
  MS_EXCEPTION_IF_NULL(primc);
  RectifyAbstract(primc, &abs_list);
  auto type = InferTypeWithAbstract(primc, abs_list);
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
      {"Reshape", [](const TM &a) { return a; }},
      {"Exp", [](const TM &a) { return exp(a); }},
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
    };
    if (func_map.find(op) == func_map.end()) {
      return nullptr;
    }
    const auto &input_a = inputs_tm[0];
    std::vector<TM> res;
    (void)std::transform(input_a.begin(), input_a.end(), std::back_inserter(res),
                         [&func_map, &op](const TM &i) { return func_map[op](i); });
    return std::make_shared<tensor::Tensor>(tid, this->shape, &res[0], tid);
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
    return std::make_shared<tensor::Tensor>(tid, this->shape, &res[0], tid);
  }
  return nullptr;
}

NodePtr PrimOp::InferValue(const NodePtrList &inputs, const DAttrs &attrs) {
  for (auto i : inputs) {
    if (i->NodeType() != NType::Value) {
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
    default:
      return nullptr;
  }
  if (res == nullptr) {
    auto [primc, inputs_abstract] = GenPrimAndAbstract(inputs, attrs);
    if (primc == nullptr) {
      return nullptr;
    }
    RectifyAbstract(primc, &inputs_abstract);
    res = InferValueWithAbstract(primc, inputs_abstract);
  }
  return res == nullptr ? nullptr : std::make_shared<ConstTensorNode>(res);
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
  auto it = std::find_if(inputs.begin(), inputs.end(),
                         [](const NodePtr &i) { return i->format != kOpFormat_DEFAULT && i->tensor_size() > 1; });
  if (it == inputs.end()) {
    return inputs.empty() ? kOpFormat_DEFAULT : inputs[0]->format;
  }
  return (*it)->format;
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
  CHECK_ATTR(attrs, "perm");
  auto perm = GetListInt(attrs.find("perm")->second);
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

NodePtr ConstantOfShapeOp::InferValue(const NodePtrList &inputs, const DAttrs &attrs) {
  for (auto i : inputs) {
    if (i->NodeType() != NType::Value) {
      return nullptr;
    }
  }
  const auto &value = GetValue<std::vector<float>>(attrs.find("value")->second);
  std::vector<float> res;
  size_t elem_num = LongToSize(std::accumulate(this->shape.begin(), this->shape.end(), 1, std::multiplies<int64_t>()));
  if (value.size() == 1) {
    res = std::vector<float>(elem_num, value[0]);
  } else if (value.size() == elem_num) {
    res = value;
  } else {
    return nullptr;
  }
  auto tensor = std::make_shared<tensor::Tensor>(this->type, this->shape, &res[0], kNumberTypeFloat32);
  return std::make_shared<ConstTensorNode>(tensor);
}

std::vector<DShape> ConstantOfShapeOp::InferShape(const NodePtrList &inputs, const DAttrs &attrs) {
  const auto &value = attrs.find("shape")->second;
  std::vector<int64_t> res;
  if (value->isa<ValueSequence>()) {
    auto valueseq = value->cast<ValueSequencePtr>();
    if (valueseq->type()->type_id() == TypeId::kNumberTypeInt32) {
      auto shp = GetValue<std::vector<int32_t>>(value);
      (void)std::transform(shp.begin(), shp.end(), std::back_inserter(res), IntToLong);
      return {res};
    } else if (valueseq->type()->type_id() == TypeId::kNumberTypeInt64) {
      res = GetValue<std::vector<int64_t>>(value);
      return {res};
    }
  } else if (value->isa<tensor::Tensor>()) {
    auto tvalue = value->cast<tensor::TensorPtr>();
    if (static_cast<TypeId>(tvalue->data_type_c()) == TypeId::kNumberTypeInt32) {
      int *data = static_cast<int *>(tvalue->data_c());
      for (size_t elem = 0; elem < tvalue->DataSize(); elem++) {
        res.push_back(IntToLong(*(data + elem)));
      }
      return {res};
    } else if (static_cast<TypeId>(tvalue->data_type_c()) == TypeId::kNumberTypeInt64) {
      int64_t *data = static_cast<int64_t *>(tvalue->data_c());
      res = std::vector<int64_t>(data, data + tvalue->DataSize());
      return {res};
    }
  }
  return PrimOp::InferShape(inputs, attrs);
}

NodePtr ShapeOp::InferValue(const NodePtrList &inputs, const DAttrs &) {
  auto tensor = std::make_shared<tensor::Tensor>(this->type, this->shape, inputs[0]->shape.data(), kNumberTypeInt64);
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

std::vector<DShape> Conv2dOp::InferShape(const NodePtrList &inputs, const DAttrs &attrs) {
  // get the output shape when format is NHWC/NCHW
  if (inputs[0]->shape.size() == 4) {
    return OpaqueOp::InferShape(inputs, attrs);
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
  if (inputs[0]->shape.size() == 4) {
    return PrimOp::InferType(inputs, attrs);
  }
  return {inputs[0]->type};
}

DFormat Conv2dOp::InferFormat(const NodePtrList &inputs, const DAttrs &attrs) {
  if (inputs[0]->shape.size() == 4) {
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
tensor::TensorPtr GatherOp::CalcGather(const NodePtrList &inputs, const DAttrs &attrs) const {
  constexpr size_t param_index = 0;
  constexpr size_t indice_index = 1;
  constexpr size_t axis_index = 2;
  constexpr size_t input_num = 3;
  constexpr size_t first_dim = 0;
  constexpr size_t second_dim = 1;
  constexpr size_t third_dim = 2;
  int64_t axis = 0;
  if (attrs.count("axis") > 0) {
    axis = GetValue<int64_t>(attrs.find("axis")->second);
  } else if (inputs.size() == input_num) {
    int *data_axis =
      static_cast<int *>(std::static_pointer_cast<inner::ConstTensorNode>(inputs[axis_index])->data()->data_c());
    axis = IntToLong(*data_axis);
  } else {
    return nullptr;
  }
  ShapeVector param_shp = inputs[param_index]->shape;
  axis = axis < 0 ? SizeToLong(param_shp.size()) + axis : axis;
  std::vector<size_t> indices;
  switch (static_cast<int>(inputs[indice_index]->type)) {
    case TypeId::kNumberTypeInt8: {
      indices = ChangeDataToVec<int8_t, size_t>(inputs[indice_index]);
      break;
    }
    case TypeId::kNumberTypeInt16: {
      indices = ChangeDataToVec<int16_t, size_t>(inputs[indice_index]);
      break;
    }
    case TypeId::kNumberTypeInt32: {
      indices = ChangeDataToVec<int32_t, size_t>(inputs[indice_index]);
      break;
    }
    case TypeId::kNumberTypeInt64: {
      indices = ChangeDataToVec<int64_t, size_t>(inputs[indice_index]);
      break;
    }
    default:
      return nullptr;
  }

  TM *input_x =
    static_cast<TM *>(std::static_pointer_cast<inner::ConstTensorNode>(inputs[param_index])->data()->data_c());
  std::vector<size_t> compact_shp = CompactShape(param_shp, axis);
  std::vector<TM> res;
  if (compact_shp.size() == input_num) {
    for (size_t i = 0; i < compact_shp[first_dim]; i++) {
      for (auto j : indices) {
        for (size_t k = 0; k < compact_shp[third_dim]; k++) {
          (void)res.emplace_back(
            input_x[i * compact_shp[second_dim] * compact_shp[third_dim] + j * compact_shp[third_dim] + k]);
        }
      }
    }
    return std::make_shared<tensor::Tensor>(this->type, this->shape, &res[0], this->type);
  }
  return nullptr;
}

NodePtr GatherOp::InferValue(const NodePtrList &inputs, const DAttrs &attrs) {
  for (auto i : inputs) {
    if (i->NodeType() != NType::Value) {
      return nullptr;
    }
  }
  TypeId output_type = this->type;
  tensor::TensorPtr res = nullptr;
  switch (static_cast<int>(output_type)) {
    case TypeId::kNumberTypeUInt8: {
      res = CalcGather<uint8_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeInt8: {
      res = CalcGather<int8_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeInt16: {
      res = CalcGather<int16_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeInt32: {
      res = CalcGather<int32_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeInt64: {
      res = CalcGather<int64_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeUInt16: {
      res = CalcGather<uint16_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeUInt32: {
      res = CalcGather<uint32_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeUInt64: {
      res = CalcGather<uint64_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeFloat16: {
      res = CalcGather<float16>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeFloat32: {
      res = CalcGather<float>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeFloat64: {
      res = CalcGather<double>(inputs, attrs);
      break;
    }
    default:
      return nullptr;
  }
  return res == nullptr ? nullptr : std::make_shared<ConstTensorNode>(res);
}

template <typename TM>
tensor::TensorPtr ConcatOp::CalcConcat(const NodePtrList &inputs, const DAttrs &attrs) {
  constexpr size_t first_dim = 0;
  constexpr size_t second_dim = 1;
  constexpr size_t third_dim = 2;
  int64_t axis = 0;
  if (attrs.count("axis") > 0) {
    axis = GetValue<int64_t>(attrs.find("axis")->second);
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
    return std::make_shared<tensor::Tensor>(this->type, this->shape, &res[0], this->type);
  }
  return nullptr;
}

NodePtr ConcatOp::InferValue(const NodePtrList &inputs, const DAttrs &attrs) {
  for (auto i : inputs) {
    if (i->NodeType() != NType::Value) {
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
    default:
      return nullptr;
  }
  return res == nullptr ? nullptr : std::make_shared<ConstTensorNode>(res);
}

std::vector<DShape> LayoutTransformOp::InferShape(const NodePtrList &inputs, const DAttrs &attrs) {
  CHECK_ATTR(attrs, kAttrSrcFormat);
  CHECK_ATTR(attrs, kAttrDstFormat);
  auto src_format = GetValue<std::string>(attrs.find(kAttrSrcFormat)->second);
  auto dst_format = GetValue<std::string>(attrs.find(kAttrDstFormat)->second);
  std::vector<int64_t> data_shape = inputs[0]->shape;
  if (src_format == kOpFormat_NHWC) {
    auto n = data_shape[0];
    auto h = data_shape[1];
    auto w = data_shape[2];
    auto c = data_shape[3];
    auto c_o_i = GkUtils::GetChannelInConvFormat(dst_format);
    if (c_o_i == 0) {
      c_o_i = 1;
    }
    auto c_o_o = c / c_o_i;
    std::vector<int64_t> output_shape{n, c_o_o, h, w, c_o_i};
    return {output_shape};
  }
  if (dst_format == kOpFormat_NHWC) {
    auto n = data_shape[0];
    auto c_o_o = data_shape[1];
    auto h = data_shape[2];
    auto w = data_shape[3];
    auto c_o_i = data_shape[4];
    auto c = c_o_o * c_o_i;
    std::vector<int64_t> output_shape{n, h, w, c};
    return {output_shape};
  }
  // LayoutTransform between nchwnc
  auto n = data_shape[0];
  auto c_o_o = data_shape[1];
  auto h = data_shape[2];
  auto w = data_shape[3];
  auto c_o_i = data_shape[4];
  auto c_o_i_new = GkUtils::GetChannelInConvFormat(dst_format);
  if (c_o_i_new == 0) {
    c_o_i_new = 1;
  }
  auto c_o_o_new = c_o_o * c_o_i / c_o_i_new;
  std::vector<int64_t> output_shape{n, c_o_o_new, h, w, c_o_i_new};
  return {output_shape};
}

std::vector<DShape> Pool2DOp::InferShape(const NodePtrList &inputs, const DAttrs &attrs) {
  CHECK_ATTR(attrs, "global");
  std::vector<int64_t> input_shape = inputs[0]->shape;
  bool is_nhwc = input_shape.size() == 4;
  int64_t n = input_shape[0];
  int64_t c, h, w;
  if (is_nhwc) {
    constexpr size_t h_idx = 1;
    constexpr size_t w_idx = 2;
    constexpr size_t c_idx = 3;
    h = input_shape[h_idx];
    w = input_shape[w_idx];
    c = input_shape[c_idx];
  } else {
    constexpr size_t c_idx = 1;
    constexpr size_t h_idx = 2;
    constexpr size_t w_idx = 3;
    c = input_shape[c_idx];
    h = input_shape[h_idx];
    w = input_shape[w_idx];
  }

  if (GetValue<bool>(attrs.find("global")->second)) {
    h = 1;
    w = 1;
  } else {
    CHECK_ATTR(attrs, "strides");
    CHECK_ATTR(attrs, "kernel_size");
    CHECK_ATTR(attrs, "round_mode");
    std::vector<int64_t> strides = GetListInt(attrs.find("strides")->second);
    std::vector<int64_t> kernels = GetListInt(attrs.find("kernel_size")->second);
    if (AnfUtils::GetIntValue(attrs.find("round_mode")->second) == 0) {
      // ceil mode
      h = ((h - kernels[0] + strides[0] - 1) / strides[0]) + 1;
      w = ((w - kernels[1] + strides[1] - 1) / strides[1]) + 1;
    } else {
      // round mode
      h = ((h - kernels[0]) / strides[0]) + 1;
      w = ((w - kernels[1]) / strides[1]) + 1;
    }
  }
  if (is_nhwc) {
    return {{n, h, w, c}};
  } else {
    auto ci = input_shape[4];
    return {{n, c, h, w, ci}};
  }
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

void StridedSliceOp::RectifyAbstract(const PrimitivePtr &primitive, AbstractBasePtrList *inputs_abstract) {
  if (!primitive->HasAttr("new_axis_mask")) {
    (void)primitive->AddAttr("new_axis_mask", MakeValue((int64_t(0))));
  }
  if (!primitive->HasAttr("shrink_axis_mask")) {
    (void)primitive->AddAttr("shrink_axis_mask", MakeValue((int64_t(0))));
  }
  if (!primitive->HasAttr("end_mask")) {
    (void)primitive->AddAttr("end_mask", MakeValue((int64_t(0))));
  }
  if (!primitive->HasAttr("begin_mask")) {
    (void)primitive->AddAttr("begin_mask", MakeValue((int64_t(0))));
  }
  if (!primitive->HasAttr("ellipsis_mask")) {
    (void)primitive->AddAttr("ellipsis_mask", MakeValue((int64_t(0))));
  }
  const mindspore::HashSet<size_t> &convert_input_list{1, 2, 3};
  auto input_names = primitive->GetAttr(kAttrInputNames);
  auto input_names_vec = GetValue<std::vector<std::string>>(input_names);
  SetAbastractsFromAttrs(primitive, convert_input_list, inputs_abstract, input_names_vec);
}

template <typename TM>
tensor::TensorPtr StridedSliceOnnxOp::CalcStridedSliceOnnx(const NodePtrList &inputs, const DAttrs &) {
  constexpr size_t input_index = 0;
  constexpr size_t begin_index = 1;
  constexpr size_t end_index = 2;
  constexpr size_t axes_index = 3;
  constexpr size_t stride_index = 4;

  ShapeVector input_shape = inputs[input_index]->shape;
  std::vector<int> begin = ChangeDataToVec<int, int>(inputs[begin_index]);
  std::vector<int> end = ChangeDataToVec<int, int>(inputs[end_index]);
  std::vector<int> axes = ChangeDataToVec<int, int>(inputs[axes_index]);
  std::vector<int> stride = ChangeDataToVec<int, int>(inputs[stride_index]);

  std::unordered_map<int, std::unordered_set<size_t>> info;
  for (size_t i = 0; i < axes.size(); i++) {
    int axis = axes[i] < 0 ? axes[i] + SizeToInt(input_shape.size()) : axes[i];
    if (begin[i] < 0 || end[i] < 0 || stride[i] < 0) {
      MS_LOG(INFO) << "Only do infervalue for StridedSliceOnnx when begin, end and stride are non-negative.";
      return nullptr;
    }
    std::unordered_set<size_t> pos;
    int index = begin[i];
    while (index < end[i]) {
      (void)pos.insert(IntToSize(index));
      index += stride[i];
    }
    (void)info.emplace(axis, pos);
  }

  TM *input_x =
    static_cast<TM *>(std::static_pointer_cast<inner::ConstTensorNode>(inputs[input_index])->data()->data_c());

  std::vector<TM> res;

  std::function<void(size_t, size_t)> func;
  func = [&func, &input_x, &res, &info, &input_shape](size_t dim, size_t offset) {
    if ((dim + 1) == input_shape.size()) {
      for (size_t i = 0; i < LongToSize(input_shape[dim]); i++) {
        if (info.count(SizeToInt(dim)) > 0) {
          if (info[SizeToInt(dim)].count(i) > 0) {
            (void)res.emplace_back(input_x[offset + i]);
          }
        } else {
          (void)res.emplace_back(input_x[offset + i]);
        }
      }
    } else if ((dim + 1) < input_shape.size()) {
      size_t accu = 1;
      for (size_t j = dim + 1; j < input_shape.size(); j++) {
        accu *= LongToSize(input_shape[j]);
      }
      for (size_t i = 0; i < LongToSize(input_shape[dim]); i++) {
        if (info.count(SizeToInt(dim)) > 0) {
          if (info[SizeToInt(dim)].count(i) > 0) {
            func(dim + 1, offset + i * accu);
          }
        } else {
          func(dim + 1, offset + i * accu);
        }
      }
    }
    return;
  };
  func(0, 0);
  return std::make_shared<tensor::Tensor>(this->type, this->shape, &res[0], this->type);
}

NodePtr StridedSliceOnnxOp::InferValue(const NodePtrList &inputs, const DAttrs &attrs) {
  for (auto i : inputs) {
    if (i->NodeType() != NType::Value) {
      return nullptr;
    }
  }
  TypeId output_type = this->type;
  tensor::TensorPtr res = nullptr;
  switch (static_cast<int>(output_type)) {
    case TypeId::kNumberTypeUInt8: {
      res = CalcStridedSliceOnnx<uint8_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeInt8: {
      res = CalcStridedSliceOnnx<int8_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeInt16: {
      res = CalcStridedSliceOnnx<int16_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeInt32: {
      res = CalcStridedSliceOnnx<int32_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeInt64: {
      res = CalcStridedSliceOnnx<int64_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeUInt16: {
      res = CalcStridedSliceOnnx<uint16_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeUInt32: {
      res = CalcStridedSliceOnnx<uint32_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeUInt64: {
      res = CalcStridedSliceOnnx<uint64_t>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeFloat16: {
      res = CalcStridedSliceOnnx<float16>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeFloat32: {
      res = CalcStridedSliceOnnx<float>(inputs, attrs);
      break;
    }
    case TypeId::kNumberTypeFloat64: {
      res = CalcStridedSliceOnnx<double>(inputs, attrs);
      break;
    }
    default:
      return nullptr;
  }
  return res == nullptr ? nullptr : std::make_shared<ConstTensorNode>(res);
}

std::vector<DShape> MatMulOp::InferShape(const NodePtrList &inputs, const DAttrs &attrs) {
  // the prim's infer shape does not supports batch dims
  constexpr size_t kMatMulRank = 2;
  if (inputs[0]->shape.size() > kMatMulRank || inputs[1]->shape.size() > kMatMulRank) {
    NodePtrList new_inputs = inputs;
    std::vector<DShape> batches(inputs.size());
    auto cut_batches = [&new_inputs, &batches, kMatMulRank](size_t i) -> void {
      const auto &shape_i = new_inputs[i]->shape;
      if (shape_i.size() > kMatMulRank) {
        DShape real_shape(shape_i.cend() - kMatMulRank, shape_i.cend());
        new_inputs[i] = std::make_shared<inner::Node>(NodeBase{real_shape, new_inputs[i]->type, new_inputs[i]->format});
        batches[i].assign(shape_i.cbegin(), shape_i.cend() - kMatMulRank);
      }
    };

    cut_batches(0);
    cut_batches(1);
    if (batches[0].size() != batches[1].size()) {
      MS_LOG(EXCEPTION) << "The Matmul's batch rank should be equal, but got " << batches[0].size() << " vs "
                        << batches[1].size();
    }
    DShape batch;
    for (size_t i = 0; i < batches[0].size(); i++) {
      if (batches[0][i] != batches[1][i]) {
        if (batches[0][i] != 1 && batches[1][i] != 1) {
          MS_LOG(EXCEPTION) << "The Matmul's batch dim is unmatched. got " << inputs[0]->shape << " and "
                            << inputs[1]->shape;
        }
      }
      batch.push_back(std::max(batches[0][i], batches[1][i]));
    }

    auto out_shape = PrimOp::InferShape(new_inputs, attrs)[0];
    // just reuse the `batch` vector
    (void)batch.insert(batch.end(), out_shape.begin(), out_shape.end());
    return {batch};
  }
  return PrimOp::InferShape(inputs, attrs);
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

void TupleGetItemOp::RectifyAbstract(const PrimitivePtr &prim, AbstractBasePtrList *abs_list) {
  if (prim->HasAttr("index")) {
    (void)abs_list->emplace_back(prim->GetAttr("index")->ToAbstract());
  }
}

void ReduceOp::RectifyAbstract(const PrimitivePtr &prim, AbstractBasePtrList *abs_list) {
  if (prim->HasAttr("axis")) {
    (void)abs_list->emplace_back(prim->GetAttr("axis")->ToAbstract());
  }
}

void ReshapeOp::RectifyAbstract(const PrimitivePtr &prim, AbstractBasePtrList *abs_list) {
  if (prim->HasAttr("shape")) {
    (void)abs_list->emplace_back(prim->GetAttr("shape")->ToAbstract());
  } else if (prim->HasAttr("input_shape")) {
    (void)abs_list->emplace_back(prim->GetAttr("input_shape")->ToAbstract());
  }
}

void TransposeOp::RectifyAbstract(const PrimitivePtr &prim, AbstractBasePtrList *abs_list) {
  if (prim->HasAttr("input_perm")) {
    (void)abs_list->emplace_back(prim->GetAttr("input_perm")->ToAbstract());
  } else if (prim->HasAttr("perm")) {
    (void)abs_list->emplace_back(prim->GetAttr("perm")->ToAbstract());
  }
}

void TileOp::RectifyAbstract(const PrimitivePtr &prim, AbstractBasePtrList *abs_list) {
  if (prim->HasAttr("multiples")) {
    (void)abs_list->emplace_back(prim->GetAttr("multiples")->ToAbstract());
  }
}

void OneHotOp::RectifyAbstract(const PrimitivePtr &prim, AbstractBasePtrList *abs_list) {
  if (prim->HasAttr("depth")) {
    (void)abs_list->emplace_back(prim->GetAttr("depth")->ToAbstract());
  }
}
}  // namespace mindspore::graphkernel::inner
