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

#include "kernel/gpu/cuda/math/broadcast_public.h"
bool IsBinaryBroadcast(const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape) {
  if (in0_shape.size() != in1_shape.size()) {
    return true;
  }
  for (size_t i = 0; i < in0_shape.size(); i++) {
    if (in0_shape[i] != in1_shape[i]) {
      return true;
    }
  }
  return false;
}

void CalSimplifyShape(const std::vector<int64_t> &aligned_in0_shape, const std::vector<int64_t> &aligned_in1_shape,
                      const std::vector<int64_t> &aligned_out_shape, std::vector<int64_t> *simplified_in0_shape,
                      std::vector<int64_t> *simplified_in1_shape, std::vector<int64_t> *simplified_out_shape) {
  // simplify shape
  simplified_in0_shape->clear();
  simplified_in1_shape->clear();
  simplified_out_shape->clear();

  auto CalStatus = [](int64_t in0_val, int64_t in1_val) -> int {
    if (in0_val == 1 || in1_val == 1) {
      if (in0_val == 1) {
        if (in1_val == 1) {
          return 0;
        } else {
          return 1;
        }
      } else {
        return 2;
      }
    } else {
      return 3;
    }
  };
  size_t head_idx = 0;
  int head_status = CalStatus(aligned_in0_shape[head_idx], aligned_in1_shape[head_idx]);
  while (head_status == 0 && head_idx < aligned_out_shape.size() - 1) {
    ++head_idx;
    head_status = CalStatus(aligned_in0_shape[head_idx], aligned_in1_shape[head_idx]);
  }
  if (head_idx == aligned_out_shape.size() - 1) {
    simplified_in0_shape->emplace_back(aligned_in0_shape.back());
    simplified_in1_shape->emplace_back(aligned_in1_shape.back());
    simplified_out_shape->emplace_back(aligned_out_shape.back());
    return;
  }
  while (head_idx < aligned_out_shape.size()) {
    int64_t in0_merged = aligned_in0_shape[head_idx];
    int64_t in1_merged = aligned_in1_shape[head_idx];
    int64_t out_merged = aligned_out_shape[head_idx];
    size_t tail_idx = head_idx + 1;
    while (tail_idx < aligned_out_shape.size()) {
      int tail_status = CalStatus(aligned_in0_shape[tail_idx], aligned_in1_shape[tail_idx]);
      if (tail_status * head_status == 0 || head_status == tail_status) {
        in0_merged *= aligned_in0_shape[tail_idx];
        in1_merged *= aligned_in1_shape[tail_idx];
        out_merged *= aligned_out_shape[tail_idx];
        ++tail_idx;
      } else {
        head_status = tail_status;
        break;
      }
    }
    head_idx = tail_idx;
    simplified_in0_shape->emplace_back(in0_merged);
    simplified_in1_shape->emplace_back(in1_merged);
    simplified_out_shape->emplace_back(out_merged);
  }
}

void SimplifyBinaryBroadcastShape(const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape,
                                  const std::vector<int64_t> &out_shape, std::vector<int64_t> *simplified_in0_shape,
                                  std::vector<int64_t> *simplified_in1_shape,
                                  std::vector<int64_t> *simplified_out_shape) {
  size_t out_rank = out_shape.size();
  size_t l_rank = in0_shape.size();
  size_t r_rank = in1_shape.size();
  size_t l_offset = out_rank - l_rank;
  std::vector<int64_t> aligned_in0_shape(in0_shape);
  std::vector<int64_t> aligned_in1_shape(in1_shape);
  std::vector<int64_t> aligned_out_shape(out_shape);
  if (aligned_in0_shape.size() == 0) {
    aligned_in0_shape.emplace_back(1);
  }
  if (aligned_in1_shape.size() == 0) {
    aligned_in1_shape.emplace_back(1);
  }
  if (aligned_out_shape.size() == 0) {
    aligned_out_shape.emplace_back(1);
  }
  // broadcast shape
  if (l_offset > 0) {
    std::vector<int64_t> insert_lft(l_offset, 1);
    aligned_in0_shape.insert(aligned_in0_shape.begin(), insert_lft.begin(), insert_lft.end());
  }
  size_t r_offset = out_rank - r_rank;
  if (r_offset > 0) {
    std::vector<int64_t> insert_rht(r_offset, 1);
    aligned_in1_shape.insert(aligned_in1_shape.begin(), insert_rht.begin(), insert_rht.end());
  }
  CalSimplifyShape(aligned_in0_shape, aligned_in1_shape, aligned_out_shape, simplified_in0_shape, simplified_in1_shape,
                   simplified_out_shape);
}

void SimplifyBroadcastToShape(const std::vector<int64_t> &inp_shape, const std::vector<int64_t> &out_shape,
                              std::vector<int64_t> *simplified_inp_shape, std::vector<int64_t> *simplified_out_shape) {
  std::vector<int64_t> aligned_inp_shape(inp_shape);
  std::vector<int64_t> aligned_out_shape(out_shape);
  if (aligned_inp_shape.size() == 0) {
    aligned_inp_shape.emplace_back(1);
  }
  if (aligned_out_shape.size() == 0) {
    aligned_out_shape.emplace_back(1);
  }
  size_t offset = aligned_out_shape.size() - aligned_inp_shape.size();
  // broadcast shape
  if (offset > 0) {
    std::vector<int64_t> insert_shape(offset, 1);
    aligned_inp_shape.insert(aligned_inp_shape.begin(), insert_shape.begin(), insert_shape.end());
  }

  // simplify shape
  simplified_inp_shape->clear();
  simplified_out_shape->clear();

  auto CalStatus = [](int64_t inp_val, int64_t out_val) -> int {
    if (inp_val == 1) {
      if (out_val == 1) {
        return 0;
      } else {
        return 1;
      }
    } else {
      return 2;
    }
  };
  size_t head_idx = 0;
  int head_status = CalStatus(aligned_inp_shape[head_idx], aligned_out_shape[head_idx]);
  while (head_status == 0 && head_idx < aligned_out_shape.size() - 1) {
    ++head_idx;
    head_status = CalStatus(aligned_inp_shape[head_idx], aligned_out_shape[head_idx]);
  }
  if (head_idx == aligned_out_shape.size() - 1) {
    simplified_inp_shape->emplace_back(aligned_inp_shape.back());
    simplified_out_shape->emplace_back(aligned_out_shape.back());
    return;
  }
  while (head_idx < aligned_out_shape.size()) {
    int64_t inp_merged = aligned_inp_shape[head_idx];
    int64_t out_merged = aligned_out_shape[head_idx];
    size_t tail_idx = head_idx + 1;
    while (tail_idx < aligned_out_shape.size()) {
      int tail_status = CalStatus(aligned_inp_shape[tail_idx], aligned_out_shape[tail_idx]);
      if (tail_status * head_status == 0 || head_status == tail_status) {
        inp_merged *= aligned_inp_shape[tail_idx];
        out_merged *= aligned_out_shape[tail_idx];
        ++tail_idx;
      } else {
        head_status = tail_status;
        break;
      }
    }
    head_idx = tail_idx;
    simplified_inp_shape->emplace_back(inp_merged);
    simplified_out_shape->emplace_back(out_merged);
  }
}
