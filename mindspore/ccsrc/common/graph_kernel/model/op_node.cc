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
#include "common/graph_kernel/model/op_node.h"

#include <math.h>
#include <algorithm>
#include <set>
#include <sstream>
#include <functional>
#include <numeric>
#include <utility>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/hash_map.h"
#include "common/graph_kernel/model/node.h"

namespace mindspore::graphkernel::inner {
std::vector<int64_t> GetListInt(const ValuePtr &attr_value) {
  bool is_int64 = true;
  auto get_int_value = [&is_int64](const ValuePtr &value) -> int64_t {
    if (value->isa<Int64Imm>()) {
      return GetValue<int64_t>(value);
    }
    is_int64 = false;
    return static_cast<int64_t>(GetValue<int>(value));
  };
  std::vector<int64_t> list_int;
  const auto &vals = attr_value->cast<ValueSequencePtr>()->value();
  (void)std::transform(vals.begin(), vals.end(), std::back_inserter(list_int), get_int_value);
  if (!is_int64) {
    MS_LOG(WARNING) << "Vector type should be 'int64_t' but got 'int'";
  }
  return list_int;
}

AbstractBasePtr InferWithAbstract(const PrimitivePtr &prim, const AbstractBasePtrList &abs_list) {
  auto &frontend_infer_func = abstract::GetPrimitiveToEvalImplMap();
  auto iter = frontend_infer_func.find(prim);
  if (iter != frontend_infer_func.end()) {
    MS_EXCEPTION_IF_NULL(iter->second.infer_shape_impl_);
    return iter->second.infer_shape_impl_(nullptr, prim, abs_list);
  }
  auto &backend_infer_func = abstract::GetPrimitiveToBackendEvalImplMap();
  auto iter2 = backend_infer_func.find(prim);
  if (iter2 != backend_infer_func.end()) {
    MS_EXCEPTION_IF_NULL(iter2->second.infer_shape_impl_);
    return iter2->second.infer_shape_impl_(nullptr, prim, abs_list);
  } else {
    MS_LOG(EXCEPTION) << "The infer function of [" << prim->name() << "] is not defined.";
  }
  return nullptr;
}

NodeBase ExtractAbstract(const AbstractBasePtr &abs) {
  NodeBase node;
  if (abs->isa<abstract::AbstractTensor>()) {
    auto shape_ptr = abs->BuildShape()->cast<abstract::ShapePtr>();
    if (shape_ptr != nullptr) {
      node.shape = shape_ptr->shape();
    }
    node.type = abs->cast<abstract::AbstractTensorPtr>()->element()->BuildType()->type_id();
  } else {  // abstract::AbstractScalar
    // leave the node.shape empty
    node.type = abs->BuildType()->type_id();
  }
  return node;
}

NodeBaseList PrimOp::InferShapeType(const NodePtrList &inputs, const DAttrs &attrs) {
  auto op_primc_fns = ops::OpPrimCRegister::GetInstance().GetPrimCMap();
  auto iter = op_primc_fns.find(op_);
  if (iter == op_primc_fns.end()) {
    MS_LOG(EXCEPTION) << "The PrimitiveC of [" << op_ << "] is not defined.";
  }
  auto primc = iter->second();
  primc->SetAttrs(attrs);
  AbstractBasePtrList inputs_abstract;
  (void)std::transform(inputs.begin(), inputs.end(), std::back_inserter(inputs_abstract),
                       [](const NodePtr &node) -> AbstractBasePtr {
                         return std::make_shared<abstract::AbstractTensor>(TypeIdToType(node->type), node->shape);
                       });
  RectifyAbstract(primc, &inputs_abstract);
  AbstractBasePtr infer_result = InferWithAbstract(primc, inputs_abstract);
  MS_EXCEPTION_IF_NULL(infer_result);
  NodeBaseList result;
  if (infer_result->isa<abstract::AbstractTuple>()) {
    for (auto abs : infer_result->cast<abstract::AbstractTuplePtr>()->elements()) {
      (void)result.emplace_back(ExtractAbstract(abs));
    }
  } else {
    (void)result.emplace_back(ExtractAbstract(infer_result));
  }
  return result;
}

NodeBaseList PrimOp::Infer(const NodePtrList &inputs, const DAttrs &attrs) {
  Check(inputs, attrs);
  NodeBaseList result;
  auto format = InferFormat(inputs, attrs);
  auto shapes = InferShape(inputs, attrs);
  auto types = InferType(inputs, attrs);
  // use PrimitiveC's inference function when InferShape or InferType returns empty result.
  if (shapes.empty() || types.empty()) {
    result = InferShapeType(inputs, attrs);
    if (!shapes.empty()) {
      if (shapes.size() != result.size()) {
        MS_LOG(EXCEPTION) << "The expected shapes num is " << result.size() << " but got " << shapes.size();
      }
      for (size_t i = 0; i < shapes.size(); i++) {
        result[i].shape = shapes[i];
      }
    }
    if (!types.empty()) {
      if (types.size() != result.size()) {
        MS_LOG(EXCEPTION) << "The expected types num is " << result.size() << " but got " << types.size();
      }
      for (size_t i = 0; i < types.size(); i++) {
        result[i].type = types[i];
      }
    }
    for (auto &r : result) {
      r.format = format;
    }
  } else {
    if (shapes.size() != types.size()) {
      MS_LOG(EXCEPTION) << "The num of shapes and types should be equal. (" << shapes.size() << " vs " << types.size()
                        << ")";
    }
    for (size_t i = 0; i < shapes.size(); i++) {
      (void)result.emplace_back(NodeBase{shapes[i], types[i], format});
    }
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
    if (i != inputs_.size() - 1) oss << ", ";
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

template <typename TM, typename TD>
tensor::TensorPtr CalcByOperator(const NodePtrList &inputs, const std::string &op, TypeId tid) {
  std::vector<TM> inputs_tm;
  (void)std::transform(inputs.begin(), inputs.end(), std::back_inserter(inputs_tm), [](const NodePtr &i) {
    return *static_cast<TM *>(std::static_pointer_cast<inner::ConstTensorNode>(i)->data()->data_c());
  });

  mindspore::HashMap<std::string, std::function<TM(const std::vector<TM> &)>> func_map = {
    {"Add", [](const std::vector<TM> &n) { return n[0] + n[1]; }},
    {"Sub", [](const std::vector<TM> &n) { return n[0] - n[1]; }},
    {"Mul", [](const std::vector<TM> &n) { return n[0] * n[1]; }},
    {"RealDiv", [](const std::vector<TM> &n) { return n[0] / n[1]; }},
    {"Neg", [](const std::vector<TM> &n) { return TM(0) - n[0]; }},
    {"Reciprocal", [](const std::vector<TM> &n) { return TM(1) / n[0]; }},
    {"Log", [](const std::vector<TM> &n) { return log(n[0]); }},
    {"Exp", [](const std::vector<TM> &n) { return exp(n[0]); }},
    {"Abs", [](const std::vector<TM> &n) { return n[0] <= TM(0) ? (TM(0) - n[0]) : n[0]; }},
    {"Sqrt", [](const std::vector<TM> &n) { return sqrt(n[0]); }},
    {"Rsqrt", [](const std::vector<TM> &n) { return TM(1) / sqrt(n[0]); }},
  };
  if (func_map.find(op) == func_map.end()) {
    return nullptr;
  }
  return std::make_shared<tensor::Tensor>(static_cast<TD>(func_map[op](inputs_tm)), TypeIdToType(tid));
}

NodePtr PrimOp::InferValue(const NodePtrList &inputs, const DAttrs &, const std::string &op) {
  for (auto i : inputs) {
    if (i->NodeType() != NType::Value) return nullptr;
  }
  TypeId output_type = this->type;
  tensor::TensorPtr res = nullptr;
  switch (static_cast<int>(output_type)) {
    case TypeId::kNumberTypeUInt8: {
      res = CalcByOperator<uint8_t, int64_t>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeInt8: {
      res = CalcByOperator<int8_t, int64_t>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeInt16: {
      res = CalcByOperator<int16_t, int64_t>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeInt32: {
      res = CalcByOperator<int32_t, int64_t>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeInt64: {
      res = CalcByOperator<int64_t, int64_t>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeUInt16: {
      res = CalcByOperator<uint16_t, int64_t>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeUInt32: {
      res = CalcByOperator<uint32_t, int64_t>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeUInt64: {
      res = CalcByOperator<uint64_t, int64_t>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeFloat16: {
      res = CalcByOperator<float16, double>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeFloat32: {
      res = CalcByOperator<float, double>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeFloat64: {
      res = CalcByOperator<double, double>(inputs, op, output_type);
      break;
    }
    default:
      return nullptr;
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
    (void)leading_shape.insert(leading_shape.end(), default_shape.begin(), default_shape.end() - SizeToLong(nz_size));
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
  (void)leading_shape.insert(leading_shape.end(), tail_shape.begin(), tail_shape.end());
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
    (void)cur.insert(cur.end(), s.begin(), s.end());
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
  if (std::all_of(inputs.begin(), inputs.end(), [](const NodePtr &input) {
        return input->format == kOpFormat_DEFAULT || input->format == kOpFormat_NHWC || input->format == kOpFormat_NCHW;
      })) {
    return PrimOp::InferShape(inputs, attrs);
  }
  if (std::all_of(inputs.begin(), inputs.end(), [](const NodePtr &input) {
        return input->format == kOpFormat_DEFAULT || input->format == kOpFormat_NHWC ||
               input->format == kOpFormat_NCHW || input->format == kOpFormat_FRAC_NZ;
      })) {
    return {BroadcastShape(inputs, true)};
  }
  std::string inputs_format;
  for (const auto &input : inputs) {
    static_cast<void>(inputs_format.append(" ").append(input->format));
  }
  MS_LOG(EXCEPTION) << "Unsupported inputs format: " << inputs_format;
}

DFormat ElemwiseOp::InferFormat(const NodePtrList &inputs, const DAttrs &) {
  auto it = std::find_if(inputs.begin(), inputs.end(), [](const NodePtr &i) { return i->format != kOpFormat_DEFAULT; });
  return it == inputs.end() ? kOpFormat_DEFAULT : (*it)->format;
}

DFormat TransposeOp::InferFormat(const NodePtrList &inputs, const DAttrs &attrs) {
  // only support NCHW/NHWC now
  if (inputs[0]->shape.size() != 4) return kOpFormat_DEFAULT;
  CHECK_ATTR(attrs, "perm");
  auto perm = GetListInt(attrs.find("perm")->second);
  const auto &ori_format = inputs[0]->format;
  if (ori_format == kOpFormat_DEFAULT || ori_format == kOpFormat_NCHW) {
    std::vector<int64_t> nchw2nhwc = {0, 2, 3, 1};
    if (perm == nchw2nhwc) return kOpFormat_NHWC;
  } else if (ori_format == kOpFormat_NHWC) {
    std::vector<int64_t> nhwc2nchw = {0, 3, 1, 2};
    if (perm == nhwc2nchw) return kOpFormat_DEFAULT;
  }
  return kOpFormat_DEFAULT;
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
}  // namespace mindspore::graphkernel::inner
