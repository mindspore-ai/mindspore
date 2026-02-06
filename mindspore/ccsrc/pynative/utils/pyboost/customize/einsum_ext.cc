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

#include "include/pynative/utils/pyboost/customize/einsum_ext.h"

#include <algorithm>
#include <string>

#include "mindspore/ops/ops_utils/op_constants.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
constexpr int64_t kIdleIdx = -1;
constexpr int64_t kLableNum = 'z' - 'a' + 1;
constexpr int64_t kTotalLabelNum = kLableNum * 2;
constexpr int64_t kEllipsisLabel = kTotalLabelNum;

TensorPtr FastPermute(const TensorPtr &input, const std::vector<int64_t> &perm) {
  int64_t dim = 0;
  for (const auto &perm_i : perm) {
    // cppcheck-suppress useStlAlgorithm
    if (perm_i != dim++) {
      return transpose(input, perm);
    }
  }
  return input;
}

void EinsumExtParseEquation(const std::string &equation_str, bool *arrow_exist,
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

  auto convert_equation_to_subscript = [](const std::string &equation_in,
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

void EinsumExtCountLabels(const std::vector<TensorPtr> &operands_list,
                          const std::vector<std::vector<int64_t>> &ops_labels, std::vector<int64_t> *labels_count,
                          int64_t *ellipsis_max_dimnum) {
  for (size_t i = 0; i < operands_list.size(); i++) {
    const auto &labels = ops_labels[i];
    auto op_rank = SizeToLong(operands_list[i]->shape().size());
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

void EinsumExtInferOutput(const std::vector<std::vector<int64_t>> &ops_labels, const std::vector<int64_t> &labels_count,
                          bool arrow_exist, int64_t ellipsis_max_dimnum, std::vector<int64_t> *labels_perm_idx,
                          int64_t *output_rank, int64_t *align_rank, int64_t *ellipsis_idx) {
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

void EinsumExtAdjustOperands(const std::vector<TensorPtr> &operands_list,
                             const std::vector<std::vector<int64_t>> &ops_labels,
                             const std::vector<int64_t> &labels_perm_idx, int64_t ellipsis_max_dimnum,
                             int64_t ellipsis_idx, int64_t align_rank, std::vector<TensorPtr> *adjust_operands,
                             std::vector<int64_t> *dim_counts) {
  auto label_to_letter = [](int64_t label) -> char {
    return label >= kLableNum ? label - kLableNum + 'a' : label + 'A';
  };
  std::vector<int64_t> no_ellipsis_dim(kTotalLabelNum, 1);
  std::vector<int64_t> ellipsis_dim(ellipsis_max_dimnum, 1);

  for (size_t i = 0; i < operands_list.size(); i++) {
    int64_t dim = 0;
    auto operand = operands_list[i];
    std::vector<int64_t> perm_axis(align_rank, kIdleIdx);
    for (const auto &label : ops_labels.at(i)) {
      if (label == kEllipsisLabel) {
        auto ellcovered_dims = SizeToLong(operands_list[i]->shape().size()) - (SizeToLong(ops_labels.at(i).size()) - 1);
        for (int64_t j = ellipsis_max_dimnum - ellcovered_dims; j < ellipsis_max_dimnum; j++) {
          if (operand->shape().at(dim) != 1) {
            if (MS_UNLIKELY(ellipsis_dim[j] != 1 && ellipsis_dim[j] != operand->shape().at(dim))) {
              MS_EXCEPTION(ValueError) << "For EinsumExt, dimension " << dim << " covered by ellipsis in " << i
                                       << " operand's shape, it is " << operand->shape().at(dim)
                                       << " that can't broadcast with dimension covered by ellipsis previously "
                                       << ellipsis_dim[j] << ".";
            }
            ellipsis_dim[j] = operand->shape().at(dim);
            (*dim_counts)[ellipsis_idx + j]++;
          }
          perm_axis[ellipsis_idx + j] = dim++;
        }
      } else if (perm_axis[labels_perm_idx[label]] == kIdleIdx) {
        if (operand->shape().at(dim) != 1) {
          if (MS_UNLIKELY(no_ellipsis_dim[label] != 1 && no_ellipsis_dim[label] != operand->shape().at(dim))) {
            MS_EXCEPTION(ValueError) << "For EinsumExt, " << label_to_letter(label) << " in operand " << i << " is "
                                     << operand->shape().at(dim) << " that can't broadcast with dimension "
                                     << no_ellipsis_dim[label] << " in previously seen operand.";
          }
          no_ellipsis_dim[label] = operand->shape().at(dim);
          (*dim_counts)[labels_perm_idx[label]]++;
        }
        perm_axis[labels_perm_idx[label]] = dim++;
      } else {
        auto dim1 = perm_axis[labels_perm_idx[label]];
        auto dim2 = dim;
        if (MS_UNLIKELY(operand->shape().at(dim1) != operand->shape().at(dim2))) {
          MS_EXCEPTION(ValueError) << "For EinsumExt, " << label_to_letter(label) << " is repeated in operand " << i
                                   << ", but it is not equal to previously seen dimension, "
                                   << operand->shape().at(dim2) << " != " << operand->shape().at(dim1) << ".";
        }
        operand = diagonal_view(operand, kIndex0, dim1, dim2);
        // movedim(-1, dim1)
        int64_t movedim_dim = 0;
        std::vector<int64_t> movedim_perm = {};
        for (int64_t k = 0; k < SizeToLong(operand->shape().size()); k++) {
          if (k == dim1) {
            movedim_perm.emplace_back(SizeToLong(operand->shape().size()) - 1);
          } else {
            movedim_perm.emplace_back(movedim_dim++);
          }
        }
        operand = FastPermute(operand, movedim_perm);
      }
    }

    for (auto &axis : perm_axis) {
      if (axis == kIdleIdx) {
        operand = expand_dims_view(operand, dim);
        axis = dim++;
      }
    }
    operand = FastPermute(operand, perm_axis);
    adjust_operands->emplace_back(operand);
  }
}

TensorPtr EinsumExtMultiplication(const TensorPtr &left_operand, const TensorPtr &right_operand,
                                  const std::vector<int64_t> &sum_dims) {
  if (sum_dims.empty()) {
    return mul(left_operand, right_operand);
  }

  auto op_sum_ext_true = [](const TensorPtr &input, const std::vector<int64_t> &dims) -> TensorPtr {
    auto dims_v = std::optional<ValueTuplePtr>(MakeValue<std::vector<int64_t>>(dims)->cast<ValueTuplePtr>());
    auto keepdim_v = std::make_shared<BoolImm>(true);
    auto dtype_v = std::nullopt;
    return sum_ext(input, dims_v, keepdim_v, dtype_v);
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

  for (int64_t i = 0; i < SizeToLong(left_operand->shape().size()); i++) {
    auto l_dim = left->shape().at(i);
    auto r_dim = right->shape().at(i);
    auto sum_l = l_dim != 1;
    auto sum_r = r_dim != 1;
    if (std::any_of(sum_dims.begin(), sum_dims.end(), [i](const auto &dim) { return i == dim; })) {
      if (sum_l && sum_r) {
        if (MS_UNLIKELY(l_dim != r_dim)) {
          MS_EXCEPTION(ValueError) << "For EinsumExt, no-broadcast dimensions to sum must be matched.";
        }
        sum_size *= l_dim;
      } else if (sum_l) {
        left = op_sum_ext_true(left, {i});
      } else if (sum_r) {
        right = op_sum_ext_true(right, {i});
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
                 [left](const auto &dim) { return left->shape().at(dim); });
  std::transform(lonly_dims.begin(), lonly_dims.end(), std::back_inserter(out_shape),
                 [left](const auto &dim) { return left->shape().at(dim); });
  std::transform(sum_dims.begin(), sum_dims.end(), std::back_inserter(out_shape), [](const auto &dim) {
    (void)(dim);
    return 1LL;
  });
  std::transform(ronly_dims.begin(), ronly_dims.end(), std::back_inserter(out_shape),
                 [right](const auto &dim) { return right->shape().at(dim); });

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

  auto op_reshape = [](const TensorPtr &input, const std::vector<int64_t> &shape) -> TensorPtr {
    return reshape(input, shape);
  };

  left = op_reshape(FastPermute(left, l_perm_axis), left_shape);
  right = op_reshape(FastPermute(right, r_perm_axis), right_shape);
  auto result = bmm_ext(left, right);
  result = FastPermute(op_reshape(result, out_shape), out_perm_axis);

  return result;
}

TensorPtr EinsumExtContractOperands(const std::vector<TensorPtr> &adjust_operands, std::vector<int64_t> dim_counts,
                                    int64_t output_rank, int64_t align_rank) {
  auto op_sum_ext = [](const TensorPtr &input, const std::vector<int64_t> &dims, bool keepdim) -> TensorPtr {
    auto dims_v = std::optional<ValueTuplePtr>(MakeValue<std::vector<int64_t>>(dims)->cast<ValueTuplePtr>());
    auto keepdim_v = std::make_shared<BoolImm>(keepdim);
    auto dtype_v = std::nullopt;
    return sum_ext(input, dims_v, keepdim_v, dtype_v);
  };
  auto result = adjust_operands[kIndex0];
  for (size_t i = kIndex1; i < adjust_operands.size(); i++) {
    std::vector<int64_t> sum_dims = {};
    std::vector<int64_t> l_dims_to_sum = {};
    std::vector<int64_t> r_dims_to_sum = {};
    auto compute_operand = adjust_operands[i];
    const auto &l_shape = result->shape();
    const auto &r_shape = compute_operand->shape();
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
      result = op_sum_ext(result, l_dims_to_sum, true);
    }
    if (!r_dims_to_sum.empty()) {
      compute_operand = op_sum_ext(compute_operand, r_dims_to_sum, true);
    }
    result = EinsumExtMultiplication(result, compute_operand, sum_dims);
  }

  if (align_rank > output_rank) {
    if (adjust_operands.size() > 1) {
      auto reshape_shape = result->shape();
      for (auto dim = align_rank - 1; dim >= output_rank; dim--) {
        reshape_shape.erase(reshape_shape.begin() + dim);
      }
      result = reshape(result, reshape_shape);
    } else {
      std::vector<int64_t> sum_dims(align_rank - output_rank);
      std::iota(sum_dims.begin(), sum_dims.end(), output_rank);
      result = op_sum_ext(result, sum_dims, false);
    }
  }

  return result;
}
}  // namespace

tensor::TensorPtr EinsumExtCustomize(const std::shared_ptr<OpRunner> &op, const StringImmPtr &equation,
                                     const ValueTuplePtr &operands) {
  auto equation_str = GetValue<std::string>(equation);
  auto operands_list = ConvertValueTupleToVector<TensorPtr>(operands);

  bool arrow_exist;
  std::vector<std::vector<int64_t>> l_ops_labels(operands_list.size());
  std::vector<std::vector<int64_t>> r_ops_labels(1);
  EinsumExtParseEquation(equation_str, &arrow_exist, &l_ops_labels, &r_ops_labels);
  // max dim num of ellipsis
  int64_t ellipsis_max_dimnum = 0;
  std::vector<int64_t> labels_count(kTotalLabelNum, 0);
  EinsumExtCountLabels(operands_list, l_ops_labels, &labels_count, &ellipsis_max_dimnum);

  int64_t align_rank = 0;
  int64_t output_rank = 0;
  int64_t ellipsis_idx = 0;
  std::vector<int64_t> labels_perm_idx(kTotalLabelNum, kIdleIdx);
  EinsumExtInferOutput(r_ops_labels, labels_count, arrow_exist, ellipsis_max_dimnum, &labels_perm_idx, &output_rank,
                       &align_rank, &ellipsis_idx);

  std::vector<TensorPtr> adjust_operands;
  std::vector<int64_t> dim_counts(align_rank, 0);
  EinsumExtAdjustOperands(operands_list, l_ops_labels, labels_perm_idx, ellipsis_max_dimnum, ellipsis_idx, align_rank,
                          &adjust_operands, &dim_counts);

  auto final_output = EinsumExtContractOperands(adjust_operands, dim_counts, output_rank, align_rank);
  op->set_outputs({final_output});
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
