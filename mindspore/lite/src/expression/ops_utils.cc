/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/expression/ops_utils.h"
#include <set>
#include <algorithm>

namespace mindspore {
namespace lite {
enum class State {
  SAME,
  X_ONE,
  Y_ONE,
};

bool CompareShape(const std::vector<int> &x_shape, const std::vector<int> &y_shape) {
  if (x_shape.size() != y_shape.size()) {
    return false;
  }

  for (size_t i = 0; i < x_shape.size(); ++i) {
    if (x_shape.at(i) != y_shape.at(i)) {
      return false;
    }
  }

  return true;
}

void ComputeReduceIndex(const std::vector<int> &reverse_x, const std::vector<int> &reverse_y,
                        std::vector<int> *grad_x_reduce_idx, std::vector<int> *grad_y_reduce_idy) {
  MS_ASSERT(grad_x_reduce_idx != nullptr);
  MS_ASSERT(grad_y_reduce_idy != nullptr);
  const size_t n = reverse_x.size();
  if (reverse_y.size() < n) {
    MS_LOG_ERROR << "The size of reverse_y is less than the size of reverse_x.";
  }
  for (size_t i = 0; i < n; ++i) {
    State curr = State::SAME;
    const int x_i = reverse_x[i];
    const int y_i = reverse_y[i];
    const int reduce_idx = (n - 1 - i);
    if (x_i == y_i) {
      curr = State::SAME;
    } else if (x_i == 1) {
      grad_x_reduce_idx->push_back(reduce_idx);
      curr = State::X_ONE;
    } else if (y_i == 1) {
      grad_y_reduce_idy->push_back(reduce_idx);
      curr = State::Y_ONE;
    } else {
      MS_LOG_ERROR << "not compatible shape input for BroadcastGradientArgs";
    }
    if (curr == State::SAME && x_i == 1) {
      grad_x_reduce_idx->push_back(reduce_idx);
      grad_y_reduce_idy->push_back(reduce_idx);
      continue;
    }
  }

  std::reverse(grad_x_reduce_idx->begin(), grad_x_reduce_idx->end());
  std::reverse(grad_y_reduce_idy->begin(), grad_y_reduce_idy->end());
}

std::vector<std::vector<int>> BroadcastGradientArgs::operator()() {
  std::vector<std::vector<int>> input_dim(kInNum);
  input_dim[0] = dim0_;
  input_dim[1] = dim1_;
  auto same_shape = CompareShape(dim0_, dim1_);
  if (same_shape) {
    return {{}, {}};
  }

  std::vector<int> reverse_x;
  std::vector<int> reverse_y;

  (void)std::transform(dim0_.rbegin(), dim0_.rend(), std::back_inserter(reverse_x), [](const int &v) { return v; });
  (void)std::transform(dim1_.rbegin(), dim1_.rend(), std::back_inserter(reverse_y), [](const int &v) { return v; });

  if (reverse_x.size() > reverse_y.size()) {
    reverse_y.resize(reverse_x.size(), 1);
  } else {
    reverse_x.resize(reverse_y.size(), 1);
  }

  std::vector<int> grad_x_reduce_idx;
  std::vector<int> grad_y_reduce_idy;
  ComputeReduceIndex(reverse_x, reverse_y, &grad_x_reduce_idx, &grad_y_reduce_idy);
  return {grad_x_reduce_idx, grad_y_reduce_idy};
}

void DynamicBroadcastGradientArgs::AddElementToGradReduceIdx(std::vector<std::vector<int>> *grad_reduce_idx,
                                                             std::vector<bool> current_is_one, bool none_is_one,
                                                             const size_t largest_rank, size_t j) {
  for (size_t i = 0; i < kInNum; ++i) {
    if (current_is_one[i] && !none_is_one) {
      (void)(*grad_reduce_idx)[i].emplace_back(largest_rank - 1 - j);
    }
  }
}

void DynamicBroadcastGradientArgs::UpdatePreIsOne(std::vector<bool> *prev_is_one, std::vector<bool> current_is_one) {
  for (size_t i = 0; i < kInNum; ++i) {
    (*prev_is_one)[i] = current_is_one[i];
  }
}

std::vector<std::vector<int>> DynamicBroadcastGradientArgs::GetGradientIndices(
  const std::vector<std::vector<int>> &reverse_shape, const size_t largest_rank) {
  std::vector<std::vector<int>> grad_reduce_idx(kInNum);
  // indices of j-th component of each input.
  std::vector<bool> prev_is_one(kInNum);
  std::vector<bool> current_is_one(kInNum);
  for (size_t i = 0; i < kInNum; ++i) {
    prev_is_one[i] = false;
    current_is_one[i] = false;
  }

  bool set_one = false;
  for (size_t j = 0; j < largest_rank; ++j) {
    int output_dim = -1;
    bool output_dim_set = false;
    bool none_is_one = true;
    // Find which indices are 1.
    for (size_t i = 0; i < kInNum; ++i) {
      if (reverse_shape[i][j] == 1) {
        current_is_one[i] = true;
        none_is_one = false;
      } else {
        current_is_one[i] = false;
        if (!output_dim_set || reverse_shape[i][j] == static_cast<int>(output_dim)) {
          output_dim = reverse_shape[i][j];
          output_dim_set = true;
        } else {
          std::cout << "Input[0] and input[1] Cannot broadcast!";
        }
      }
    }
    // All dimensions are 1.
    if (!output_dim_set) {
      for (size_t i = 0; i < kInNum; ++i) {
        (void)grad_reduce_idx[i].emplace_back(largest_rank - 1 - j);
      }
      continue;
    } else if (std::equal(current_is_one.begin(), current_is_one.end(), prev_is_one.begin()) && set_one) {
      AddElementToGradReduceIdx(&grad_reduce_idx, current_is_one, none_is_one, largest_rank, j);
    } else {
      AddElementToGradReduceIdx(&grad_reduce_idx, current_is_one, none_is_one, largest_rank, j);
    }
    set_one = true;
    UpdatePreIsOne(&prev_is_one, current_is_one);
  }
  return grad_reduce_idx;
}

std::vector<std::vector<int>> DynamicBroadcastGradientArgs::CalculateOutput(const std::vector<std::vector<int>> &x) {
  std::vector<std::vector<int>> grad_reduce_idx(kInNum);
  bool all_equal = true;
  size_t largest_rank = 0;
  for (size_t i = 0; i < kInNum; ++i) {
    if (x[i] != x[0]) {
      all_equal = false;
    }
    if (x[i].size() > largest_rank) {
      largest_rank = x[i].size();
    }
  }
  if (all_equal) {
    return grad_reduce_idx;
  }

  // Reverse input the shapes
  std::vector<std::vector<int>> reverse_shape(kInNum);
  for (size_t i = 0; i < kInNum; ++i) {
    reverse_shape[i] = x[i];
    std::reverse(reverse_shape[i].begin(), reverse_shape[i].end());
  }

  // 1-extend and align all vectors.
  for (size_t i = 0; i < kInNum; ++i) {
    if (reverse_shape[i].size() < largest_rank) {
      reverse_shape[i].resize(largest_rank, 1);
    }
  }
  grad_reduce_idx = GetGradientIndices(reverse_shape, largest_rank);
  return grad_reduce_idx;
}

std::vector<std::vector<int>> DynamicBroadcastGradientArgs::SetOutputValue(
  const std::vector<std::vector<int>> &grad_reduce_idx, const std::vector<std::vector<int>> &input_dim) {
  std::vector<std::vector<int>> output(kInNum);
  for (size_t index = 0; index < kInNum; ++index) {
    auto idx_num = grad_reduce_idx[index].size();
    for (size_t k = 0; k < idx_num; ++k) {
      output[index].push_back(grad_reduce_idx[index][idx_num - 1 - k]);
    }
    if (idx_num == 0) {
      auto input_num = input_dim[index].size();
      for (size_t k = 0; k < input_num; ++k) {
        output[index].push_back(k);
      }
    }
  }
  return output;
}

std::vector<std::vector<int>> DynamicBroadcastGradientArgs::operator()() {
  std::vector<std::vector<int>> input_dim(kInNum);
  input_dim[0] = dim0_;
  input_dim[1] = dim1_;
  auto grad_reduce_idx = CalculateOutput(input_dim);
  auto output = SetOutputValue(grad_reduce_idx, input_dim);
  return output;
}

std::vector<int> VectorDiv::operator()(const std::vector<int> &x, const std::vector<int> &d) {
  if (d.size() != x.size()) {
    MS_LOG(ERROR) << "x and divider must have same size";
    return {};
  }
  std::vector<int> res;
  for (size_t i = 0; i < d.size(); i++) {
    auto x_value = x.at(i);
    auto d_value = d.at(i);
    if (d_value == 0) {
      MS_LOG(ERROR) << "Divisor is zero";
      return {};
    }
    if ((x_value % d_value) != 0) {
      MS_LOG(ERROR) << "x and d and not dividable";
    }
    auto r = x_value / d_value;
    res.push_back(r);
  }
  return res;
}

std::vector<int> ShapeReduce::operator()(const std::vector<int> &x_shape, const std::vector<int> &axis) {
  int x_rank = x_shape.size();
  std::set<int> axis_set;

  auto min = -x_rank;
  auto max = x_rank - 1;
  for (auto &elem : axis) {
    if (elem > max || elem < min) {
      MS_LOG(ERROR) << "illegal axis value";
      return {};
    }
    axis_set.insert(elem);
  }
  std::vector<int> res;
  for (int i = 0; i < x_rank; i++) {
    if (axis_set.count(i) || axis_set.count(i - x_rank)) {
      res.push_back(1);
    } else {
      res.push_back(x_shape.at(i));
    }
  }
  return res;
}
}  // namespace lite
}  // namespace mindspore
