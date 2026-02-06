/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "mindspore/ccsrc/frontend/operator/meta_dsl/func_op/einsum_ext.h"
#include <algorithm>
#include <string>
#include <utility>
#include "ir/dtype/type.h"
#include "utils/shape_utils.h"
#include "ops/op_def_utils.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_constants.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_e.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore::prim {
namespace {
constexpr int64_t kIdleIdx = -1;
constexpr int64_t kLableNum = 'z' - 'a' + 1;
constexpr int64_t kTotalLabelNum = kLableNum * 2;
constexpr int64_t kEllipsisLabel = kTotalLabelNum;
void EinsumExtFuncOpParseEquation(const std::string &equation_str, bool *arrow_exist,
                                  std::vector<std::vector<int64_t>> *l_ops_labels,
                                  std::vector<std::vector<int64_t>> *r_ops_labels) {
  std::string equation = equation_str;
  const std::string arrow = "->";
  auto arrow_pos = equation.find(arrow);
  *arrow_exist = arrow_pos != std::string::npos;

  std::string l_equation;
  std::string r_equation;
  if (*arrow_exist) {
    l_equation = equation.substr(0, arrow_pos);
    r_equation = equation.substr(arrow_pos + arrow.length());
  } else {
    l_equation = equation;
  }

  auto convert_equation_to_subscript = [](const std::string equation_in,
                                          std::vector<std::vector<int64_t>> *ops_labels) -> void {
    auto letter_to_subscript = [](unsigned char letter) -> int64_t {
      if (letter == '.') {
        return kEllipsisLabel;
      }
      if (!std::isalpha(letter)) {
        MS_EXCEPTION(ValueError) << "For EinsumExt, the letter in equation must be in [a-zA-Z] or ., but got "
                                 << letter;
      }
      return std::isupper(letter) ? letter - 'A' : letter - 'a' + kLableNum;
    };

    constexpr auto kEllOffset = 2;
    constexpr auto kDot = '.';
    size_t curr_op = 0;
    bool find_ellipsis = false;
    for (size_t i = 0; i < equation_in.length(); i++) {
      const auto label = equation_in.at(i);
      switch (label) {
        case ' ':
          break;

        case '.': {
          bool is_valid_ellipsis =
            (i + kEllOffset) < equation_in.length() && equation_in.at(++i) == kDot && equation_in.at(++i) == kDot;
          if (MS_UNLIKELY(find_ellipsis || !is_valid_ellipsis)) {
            MS_EXCEPTION(ValueError) << "For EinsumExt, an ellipsis must include three continuous dot, "
                                     << "and can only be found once in left and right equation.";
          }
          (*ops_labels)[curr_op].emplace_back(letter_to_subscript(label));
          find_ellipsis = true;
          break;
        }

        case ',': {
          if (MS_UNLIKELY(++curr_op >= ops_labels->size())) {
            MS_EXCEPTION(ValueError) << "For EinsumExt, the number of operands is fewer than specified in equation.";
          }
          find_ellipsis = false;
          break;
        }

        default: {
          (*ops_labels)[curr_op].emplace_back(letter_to_subscript(label));
          break;
        }
      }
    }
    if (MS_UNLIKELY((curr_op + 1) != ops_labels->size())) {
      MS_EXCEPTION(ValueError) << "For EinsumExt, the number of operands is not same as specified in equation.";
    }
  };

  convert_equation_to_subscript(l_equation, l_ops_labels);
  convert_equation_to_subscript(r_equation, r_ops_labels);
}

void EinsumExtFuncOpCountLabels(const ShapeArray &operands_shape, const std::vector<std::vector<int64_t>> &ops_labels,
                                std::vector<int64_t> *labels_count, int64_t *ellipsis_max_dimnum) {
  for (size_t i = 0; i < operands_shape.size(); i++) {
    const auto &labels = ops_labels[i];
    const auto &shape = operands_shape[i];
    auto op_rank = SizeToLong(shape.size());
    auto labels_num = SizeToLong(labels.size());
    bool find_ellipsis = false;

    for (const auto &label : labels) {
      if (label == kEllipsisLabel) {
        labels_num--;
        *ellipsis_max_dimnum = std::max(*ellipsis_max_dimnum, op_rank - labels_num);
        find_ellipsis = true;
      } else {
        (*labels_count)[label]++;
      }
    }

    if (MS_UNLIKELY(find_ellipsis && (labels_num > op_rank))) {
      MS_EXCEPTION(ValueError) << "For EinsumExt, the number of labels in " << i
                               << " sub-equation is more than the rank of operand.";
    }

    if (MS_UNLIKELY(!find_ellipsis && (labels_num != op_rank))) {
      MS_EXCEPTION(ValueError) << "For EinsumExt, the number of labels in " << i
                               << " sub-equation is not equal to the rank of operand, but got " << labels_num
                               << " labels and " << op_rank << " dimensions.";
    }
  }
}

void EinsumExtFuncOpInferOutput(const std::vector<std::vector<int64_t>> &ops_labels,
                                const std::vector<int64_t> &labels_count, bool arrow_exist, int64_t ellipsis_max_dimnum,
                                std::vector<int64_t> *labels_perm_idx, int64_t *output_rank, int64_t *align_rank,
                                int64_t *ellipsis_idx) {
  int64_t perm_idx = 0;
  bool find_ellipsis = false;
  if (arrow_exist) {
    auto label_to_letter = [](int64_t label) -> char {
      return label >= kLableNum ? label - kLableNum + 'a' : label + 'A';
    };

    for (const auto &label : ops_labels[0]) {
      if (label == kEllipsisLabel) {
        *ellipsis_idx = perm_idx;
        perm_idx += ellipsis_max_dimnum;
        find_ellipsis = true;
      } else {
        if (MS_UNLIKELY((*labels_perm_idx)[label] != kIdleIdx)) {
          MS_EXCEPTION(ValueError) << "For EinsumExt, " << label_to_letter(label)
                                   << " has appeared more than once in output equation.";
        }
        if (MS_UNLIKELY(labels_count[label] == 0)) {
          MS_EXCEPTION(ValueError) << "For EinsumExt, " << label_to_letter(label)
                                   << " does not appear in any input equations.";
        }
        (*labels_perm_idx)[label] = perm_idx++;
      }
    }
  } else {
    perm_idx = ellipsis_max_dimnum;
    find_ellipsis = true;
    for (size_t label = 0; label < kTotalLabelNum; label++) {
      if (labels_count[label] != 1) {
        continue;
      }
      (*labels_perm_idx)[label] = perm_idx++;
    }
  }

  *output_rank = perm_idx;
  if (!find_ellipsis) {
    *ellipsis_idx = perm_idx;
    perm_idx += ellipsis_max_dimnum;
  }

  for (size_t label = 0; label < kTotalLabelNum; label++) {
    if (labels_count[label] > 0 && (*labels_perm_idx)[label] == kIdleIdx) {
      (*labels_perm_idx)[label] = perm_idx++;
    }
  }
  *align_rank = perm_idx;
}

void GetTransposeOutShape(ShapeVector *shape, const std::vector<int64_t> &perm) {
  auto ori_shape = *shape;
  for (size_t dim = 0; dim < ori_shape.size(); dim++) {
    (*shape)[dim] = ori_shape[perm[dim]];
  }
}
}  // namespace

void CheckEinsumExtInputs(const PrimitivePtr &primitive, const AbstractBasePtrList &input_args) {
  auto equation_opt = GetScalarValue<std::string>(input_args[kIndex0]->GetValue());
  if (MS_UNLIKELY(!equation_opt.has_value())) {
    MS_EXCEPTION(ValueError) << "For EinsumExt, equation must be constant string.";
  }

  auto operands = input_args[kIndex1];
  if (MS_UNLIKELY(operands->GetShape()->isa<abstract::DynamicSequenceShape>())) {
    MS_EXCEPTION(ValueError) << "For EinsumExt, operands can't be a tuple with dynamic length.";
  }

  auto tuple_shape = operands->GetShape()->cast<abstract::SequenceShapePtr>();
  MS_EXCEPTION_IF_NULL(tuple_shape);
  auto shapes = tuple_shape->shape();
  if (MS_UNLIKELY(shapes.empty())) {
    MS_EXCEPTION(ValueError) << "For EinsumExt, operands can't be empty.";
  }

  ShapeArray operands_shape = {};
  for (auto &shape : shapes) {
    auto tensor_shape = shape->cast<abstract::TensorShapePtr>();
    MS_EXCEPTION_IF_NULL(tensor_shape);
    auto shape_vector = tensor_shape->GetShapeVector();
    if (MS_UNLIKELY(IsDynamic(shape_vector))) {
      MS_EXCEPTION(ValueError) << "For EinsumExt, dynamic shape or dynamic rank is not supported yet, but got shape "
                               << ShapeVectorToStr(shape_vector) << " in operands.";
    }
    operands_shape.emplace_back(std::move(shape_vector));
  }
  primitive->AddAttr("equation_str", MakeValue(equation_opt.value()));
  primitive->AddAttr("operands_shape", MakeValue(operands_shape));
}

NodePtr EinsumExtMetaImpl::FastPermute(const NodePtr &input, const std::vector<int64_t> &perm) {
  int64_t dim = 0;
  for (const auto &perm_i : perm) {
    if (perm_i != dim++) {
      return Call(Prim(TransposeView), input, Value(perm));
    }
  }
  return input;
}

void EinsumExtMetaImpl::AdjustOperands(const std::vector<NodePtr> &operands_list,
                                       const std::vector<std::vector<int64_t>> &ops_labels,
                                       const std::vector<int64_t> &labels_perm_idx, int64_t ellipsis_max_dimnum,
                                       int64_t ellipsis_idx, int64_t align_rank, ShapeArray *operands_shape,
                                       std::vector<NodePtr> *adjust_operands, std::vector<int64_t> *dim_counts) {
  auto label_to_letter = [](int64_t label) -> char {
    return label >= kLableNum ? label - kLableNum + 'a' : label + 'A';
  };
  std::vector<int64_t> no_ellipsis_dim(kTotalLabelNum, 1);
  std::vector<int64_t> ellipsis_dim(ellipsis_max_dimnum, 1);

  for (size_t i = 0; i < operands_list.size(); i++) {
    int64_t dim = 0;
    auto operand = operands_list[i];
    auto &shape = (*operands_shape)[i];
    auto rank = (*operands_shape)[i].size();
    std::vector<int64_t> perm_axis(align_rank, kIdleIdx);
    for (const auto &label : ops_labels.at(i)) {
      if (label == kEllipsisLabel) {
        auto ellcovered_dims = SizeToLong(rank) - (SizeToLong(ops_labels.at(i).size()) - 1);
        for (int64_t j = ellipsis_max_dimnum - ellcovered_dims; j < ellipsis_max_dimnum; j++) {
          if (shape.at(dim) != 1) {
            if (MS_UNLIKELY(ellipsis_dim[j] != 1 && ellipsis_dim[j] != shape.at(dim))) {
              MS_EXCEPTION(ValueError) << "For EinsumExt, dimension " << dim << " covered by ellipsis in " << i
                                       << " operand's shape, it is " << shape.at(dim)
                                       << " that can't broadcast with dimension covered by ellipsis previously "
                                       << ellipsis_dim[j] << ".";
            }
            ellipsis_dim[j] = shape.at(dim);
            (*dim_counts)[ellipsis_idx + j]++;
          }
          perm_axis[ellipsis_idx + j] = dim++;
        }
      } else if (perm_axis[labels_perm_idx[label]] == kIdleIdx) {
        if (shape.at(dim) != 1) {
          if (MS_UNLIKELY(no_ellipsis_dim[label] != 1 && no_ellipsis_dim[label] != shape.at(dim))) {
            MS_EXCEPTION(ValueError) << "For EinsumExt, " << label_to_letter(label) << " in operand " << i << " is "
                                     << shape.at(dim) << " that can't broadcast with dimension "
                                     << no_ellipsis_dim[label] << " in previously seen operand.";
          }
          no_ellipsis_dim[label] = shape.at(dim);
          (*dim_counts)[labels_perm_idx[label]]++;
        }
        perm_axis[labels_perm_idx[label]] = dim++;
      } else {
        auto dim1 = perm_axis[labels_perm_idx[label]];
        auto dim2 = dim;
        if (MS_UNLIKELY(shape.at(dim1) != shape.at(dim2))) {
          MS_EXCEPTION(ValueError) << "For EinsumExt, " << label_to_letter(label) << " is repeated in operand " << i
                                   << ", but it is not equal to previously seen dimension, " << shape.at(dim2)
                                   << " != " << shape.at(dim1) << ".";
        }
        operand = Call(Prim(DiagonalView), operand, Value(0), Value(dim1), Value(dim2));
        auto diagonal_rank = SizeToLong(shape.size()) - 1;
        int64_t movedim_dim = 0;
        std::vector<int64_t> movedim_perm = {};
        for (int64_t k = 0; k < diagonal_rank; k++) {
          if (k == dim1) {
            movedim_perm.emplace_back(diagonal_rank - 1);
          } else {
            movedim_perm.emplace_back(movedim_dim++);
          }
        }
        operand = FastPermute(operand, movedim_perm);
        shape.erase(shape.begin() + dim2);
      }
    }

    for (auto &axis : perm_axis) {
      if (axis == kIdleIdx) {
        operand = Call(Prim(ExpandDims), operand, Value(dim));
        shape.insert(shape.begin() + dim, 1LL);
        axis = dim++;
      }
    }
    operand = FastPermute(operand, perm_axis);
    GetTransposeOutShape(&shape, perm_axis);
    (*adjust_operands).emplace_back(operand);
  }
}

NodePtr EinsumExtMetaImpl::Multiplication(const NodePtr &left_operand, const NodePtr &right_operand,
                                          const std::vector<int64_t> &sum_dims, ShapeVector *l_shape,
                                          ShapeVector r_shape) {
  if (sum_dims.empty()) {
    for (size_t i = 0; i < (*l_shape).size(); i++) {
      (*l_shape)[i] = std::max((*l_shape)[i], r_shape[i]);
    }
    return Call(Prim(Mul), left_operand, right_operand);
  }

  auto op_sum_ext_true = [this](const NodePtr &input, const std::vector<int64_t> &dims, ShapeVector *shape) -> NodePtr {
    for (size_t i = 0; i < dims.size(); i++) {
      (*shape)[dims[i]] = 1;
    }
    return this->Call(Prim(SumExt), input, this->Value(dims), this->Value(true), this->Value(kNone));
  };
  std::vector<int64_t> batch_dims;
  std::vector<int64_t> lonly_dims;
  std::vector<int64_t> ronly_dims;
  int64_t batch_size = 1;
  int64_t lonly_size = 1;
  int64_t ronly_size = 1;
  int64_t sum_size = 1;
  auto left = left_operand;
  auto right = right_operand;
  auto rank = SizeToLong(l_shape->size());

  for (int64_t i = 0; i < rank; i++) {
    auto l_dim = (*l_shape)[i];
    auto r_dim = r_shape[i];
    auto sum_l = l_dim != 1;
    auto sum_r = r_dim != 1;
    if (std::any_of(sum_dims.begin(), sum_dims.end(), [i](const auto &dim) { return i == dim; })) {
      if (sum_l && sum_r) {
        if (MS_UNLIKELY(l_dim != r_dim)) {
          MS_EXCEPTION(ValueError) << "For EinsumExt, no-broadcast dimensions to sum must be matched.";
        }
        sum_size *= l_dim;
      } else if (sum_l) {
        left = op_sum_ext_true(left, {i}, l_shape);
      } else if (sum_r) {
        right = op_sum_ext_true(right, {i}, &r_shape);
      }
    } else if (sum_l && sum_r) {
      if (MS_UNLIKELY(l_dim != r_dim)) {
        MS_EXCEPTION(ValueError) << "For EinsumExt, no-broadcast dimensions to sum must be matched.";
      }
      batch_dims.emplace_back(i);
      batch_size *= l_dim;
    } else if (sum_l) {
      lonly_dims.emplace_back(i);
      lonly_size *= l_dim;
    } else {
      ronly_dims.emplace_back(i);
      ronly_size *= r_dim;
    }
  }

  auto out_rank = batch_dims.size() + lonly_dims.size() + sum_dims.size() + ronly_dims.size();
  std::vector<int64_t> left_shape = {batch_size, lonly_size, sum_size};
  std::vector<int64_t> right_shape = {batch_size, sum_size, ronly_size};
  std::vector<int64_t> out_shape;
  out_shape.reserve(out_rank);
  std::transform(batch_dims.begin(), batch_dims.end(), std::back_inserter(out_shape),
                 [l_shape](const auto &dim) { return (*l_shape)[dim]; });
  std::transform(lonly_dims.begin(), lonly_dims.end(), std::back_inserter(out_shape),
                 [l_shape](const auto &dim) { return (*l_shape)[dim]; });
  std::transform(sum_dims.begin(), sum_dims.end(), std::back_inserter(out_shape), [](const auto &dim) {
    (void)(dim);
    return 1LL;
  });
  std::transform(ronly_dims.begin(), ronly_dims.end(), std::back_inserter(out_shape),
                 [r_shape](const auto &dim) { return r_shape[dim]; });

  std::vector<int64_t> l_perm_axis(batch_dims);
  l_perm_axis.insert(l_perm_axis.end(), lonly_dims.begin(), lonly_dims.end());
  l_perm_axis.insert(l_perm_axis.end(), sum_dims.begin(), sum_dims.end());
  l_perm_axis.insert(l_perm_axis.end(), ronly_dims.begin(), ronly_dims.end());

  std::vector<int64_t> r_perm_axis(batch_dims);
  r_perm_axis.insert(r_perm_axis.end(), sum_dims.begin(), sum_dims.end());
  r_perm_axis.insert(r_perm_axis.end(), ronly_dims.begin(), ronly_dims.end());
  r_perm_axis.insert(r_perm_axis.end(), lonly_dims.begin(), lonly_dims.end());

  std::vector<int64_t> out_perm_axis(out_rank, -1);
  int64_t out_dim = 0;
  for (size_t dim = 0; dim < l_perm_axis.size(); dim++) {
    out_perm_axis[l_perm_axis[dim]] = out_dim++;
  }

  auto op_reshape = [this](const NodePtr &input, const std::vector<int64_t> &shape) -> NodePtr {
    return this->Call(Prim(Reshape), input, this->Value(shape));
  };

  left = op_reshape(FastPermute(left, l_perm_axis), left_shape);
  right = op_reshape(FastPermute(right, r_perm_axis), right_shape);
  auto result = Call(Prim(BatchMatMulExt), left, right);
  result = FastPermute(op_reshape(result, out_shape), out_perm_axis);

  *l_shape = out_shape;
  for (size_t dim = 0; dim < out_shape.size(); dim++) {
    (*l_shape)[dim] = out_shape[out_perm_axis[dim]];
  }

  return result;
}

NodePtr EinsumExtMetaImpl::ContractOperands(const std::vector<NodePtr> &adjust_operands,
                                            const ShapeArray &operands_shape, std::vector<int64_t> dim_counts,
                                            int64_t output_rank, int64_t align_rank) {
  auto op_sum_ext = [this](const NodePtr &input, std::vector<int64_t> dims, bool keepdim,
                           ShapeVector *shape) -> NodePtr {
    if (keepdim) {
      for (size_t i = 0; i < dims.size(); i++) {
        (*shape)[dims[i]] = 1;
      }
    } else {
      std::sort(dims.begin(), dims.end());
      std::vector<int64_t>::reverse_iterator it_re;
      for (it_re = dims.rbegin(); it_re != dims.rend(); ++it_re) {
        (void)(*shape).erase((*shape).begin() + *it_re);
      }
    }
    return this->Call(Prim(SumExt), input, this->Value(dims), this->Value(keepdim), this->Value(kNone));
  };
  auto result = adjust_operands[kIndex0];
  auto l_shape = operands_shape[kIndex0];
  for (size_t i = kIndex1; i < adjust_operands.size(); i++) {
    std::vector<int64_t> sum_dims = {};
    std::vector<int64_t> l_dims_to_sum = {};
    std::vector<int64_t> r_dims_to_sum = {};
    auto compute_operand = adjust_operands[i];
    auto r_shape = operands_shape[i];
    for (int64_t dim = output_rank; dim < align_rank; dim++) {
      if (l_shape[dim] != 1 && r_shape[dim] != 1) {
        if (--dim_counts[dim] == 1) {
          sum_dims.emplace_back(dim);
          dim_counts[dim] = 0;
        }
      } else if (dim_counts[dim] == 1) {
        if (l_shape[dim] != 1) {
          l_dims_to_sum.emplace_back(dim);
          dim_counts[dim] = 0;
        } else if (r_shape[dim] != 1) {
          r_dims_to_sum.emplace_back(dim);
          dim_counts[dim] = 0;
        }
      }
    }
    if (!l_dims_to_sum.empty()) {
      result = op_sum_ext(result, l_dims_to_sum, true, &l_shape);
    }
    if (!r_dims_to_sum.empty()) {
      compute_operand = op_sum_ext(compute_operand, r_dims_to_sum, true, &r_shape);
    }
    result = Multiplication(result, compute_operand, sum_dims, &l_shape, r_shape);
  }

  if (align_rank > output_rank) {
    if (adjust_operands.size() > 1) {
      for (auto dim = align_rank - 1; dim >= output_rank; dim--) {
        l_shape.erase(l_shape.begin() + dim);
      }
      result = Call(Prim(Reshape), result, Value(l_shape));
    } else {
      std::vector<int64_t> sum_dims(align_rank - output_rank);
      std::iota(sum_dims.begin(), sum_dims.end(), output_rank);
      result = op_sum_ext(result, sum_dims, false, &l_shape);
    }
  }

  return result;
}

BeginFunction(EinsumExt, equation, operands) {
  const auto &equation_str_v = prim()->GetAttr("equation_str");
  const auto &operands_shape_v = prim()->GetAttr("operands_shape");
  MS_EXCEPTION_IF_NULL(equation_str_v);
  MS_EXCEPTION_IF_NULL(operands_shape_v);
  auto equation_str = GetValue<std::string>(equation_str_v);
  auto operands_shape = GetValue<ShapeArray>(operands_shape_v);

  bool arrow_exist;
  std::vector<std::vector<int64_t>> l_ops_labels(operands_shape.size());
  std::vector<std::vector<int64_t>> r_ops_labels(1);
  EinsumExtFuncOpParseEquation(equation_str, &arrow_exist, &l_ops_labels, &r_ops_labels);

  // max dim num of ellipsis
  int64_t ellipsis_max_dimnum = 0;
  std::vector<int64_t> labels_count(kTotalLabelNum, 0);
  EinsumExtFuncOpCountLabels(operands_shape, l_ops_labels, &labels_count, &ellipsis_max_dimnum);

  int64_t align_rank = 0;
  int64_t output_rank = 0;
  int64_t ellipsis_idx = 0;
  std::vector<int64_t> labels_perm_idx(kTotalLabelNum, kIdleIdx);
  EinsumExtFuncOpInferOutput(r_ops_labels, labels_count, arrow_exist, ellipsis_max_dimnum, &labels_perm_idx,
                             &output_rank, &align_rank, &ellipsis_idx);

  std::vector<NodePtr> operands_list;
  operands_list.reserve(operands_shape.size());
  for (int64_t i = 0; i < SizeToLong(operands_shape.size()); i++) {
    operands_list.emplace_back(GetItem(operands, Value(i)));
  }

  std::vector<NodePtr> adjust_operands;
  std::vector<int64_t> dim_counts(align_rank, 0);
  AdjustOperands(operands_list, l_ops_labels, labels_perm_idx, ellipsis_max_dimnum, ellipsis_idx, align_rank,
                 &operands_shape, &adjust_operands, &dim_counts);

  auto final_output = ContractOperands(adjust_operands, operands_shape, dim_counts, output_rank, align_rank);
  Return(final_output);
}
EndFunction(EinsumExt)
}  // namespace mindspore::prim
