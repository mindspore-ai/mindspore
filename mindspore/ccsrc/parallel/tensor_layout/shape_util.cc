/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "parallel/tensor_layout/shape_util.h"
#include <utility>
#include "parallel/status.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {

/*
 * example:
 * shape = [2, 8, 32]
 * shape_accum = [2, 2 * 8, 2 * 8 * 32]
 */
Status ShapeToAccumulateProduct(const std::vector<int32_t>& shape, std::vector<int64_t>* shape_accum) {
  MS_EXCEPTION_IF_NULL(shape_accum);
  shape_accum->clear();
  int64_t size = 1;
  for (auto iter = shape.begin(); iter < shape.end(); ++iter) {
    size *= *iter;
    if (size <= 0) {
      MS_LOG(ERROR) << "element of shape should not be zero";
      return Status::FAILED;
    }
    shape_accum->push_back(size);
  }
  return Status::SUCCESS;
}

/*
 * example:
 * shape = [2, 8, 32]
 * shape_accum = [2 * 8 * 32, 8 * 32, 32]
 *
 */
Status ShapeToAccumulateProductReverse(const std::vector<int32_t>& shape, std::vector<int64_t>* shape_accum) {
  MS_EXCEPTION_IF_NULL(shape_accum);
  shape_accum->clear();
  int64_t size = 1;
  for (auto iter = shape.end() - 1; iter >= shape.begin(); --iter) {
    size *= *iter;
    if (size <= 0) {
      MS_LOG(ERROR) << "element of shape should not be zero";
      return Status::FAILED;
    }
    (void)shape_accum->insert(shape_accum->begin(), size);
  }
  return Status::SUCCESS;
}

/*
 * example:
 * shape_accum = [2, 2 * 8, 2 * 8 * 32]
 * shape = [2, 8, 32]
 *
 */
Status AccumulateProductToShape(const std::vector<int64_t>& shape_accum, std::vector<int32_t>* shape) {
  MS_EXCEPTION_IF_NULL(shape);
  shape->clear();
  int64_t value = 1;
  for (auto iter = shape_accum.begin(); iter < shape_accum.end(); ++iter) {
    if ((*iter) == 0) {
      MS_LOG(ERROR) << "element of shape_accum should not be zero";
      return Status::FAILED;
    }
    if ((*iter) % value != 0) {
      MS_LOG(ERROR) << "shape_accum is not a accumulate product in ascending order";
      return Status::FAILED;
    }
    shape->push_back(static_cast<int32_t>((*iter) / value));
    value = (*iter);
  }
  return Status::SUCCESS;
}

/*
 * example:
 * shape_accum_reverse = [2 * 8 * 32, 8 * 32, 32]
 * shape = [2, 8, 32]
 */
Status AccumulateProductReverseToShape(const std::vector<int64_t>& shape_accum_reverse, std::vector<int32_t>* shape) {
  MS_EXCEPTION_IF_NULL(shape);
  shape->clear();
  int64_t value = 1;
  for (auto iter = shape_accum_reverse.end() - 1; iter >= shape_accum_reverse.begin(); --iter) {
    if (*iter == 0) {
      MS_LOG(ERROR) << "element of shape_accum should not be zero";
      return Status::FAILED;
    }
    if ((*iter) % value != 0) {
      MS_LOG(ERROR) << "shape_accum is not a accumulate product in ascending order";
      return Status::FAILED;
    }
    (void)shape->insert(shape->begin(), static_cast<int32_t>((*iter) / value));
    value = *iter;
  }
  return Status::SUCCESS;
}

/*
 * example1:
 * in1 = [2, 8]
 * in2 = [4, 8]
 * *out = [2, 4, 8]
 *
 * example2:
 * in1 = [2, 4, 16]
 * in2 = [8, 16]
 * *out = [2, 4, 8, 16]
 */
Status UnifyAccumulateProduct(const std::vector<int64_t>& in1_accum, const std::vector<int64_t>& in2_accum,
                              std::vector<int64_t>* out_accum) {
  MS_EXCEPTION_IF_NULL(out_accum);
  out_accum->clear();
  auto in1_iter = in1_accum.begin();
  auto in2_iter = in2_accum.begin();
  while ((in1_iter < in1_accum.end()) || (in2_iter < in2_accum.end())) {
    if ((*in1_iter <= 0) || (*in2_iter <= 0)) {
      MS_LOG(ERROR) << "element of in1 and in2 must be larger than zero";
      return Status::FAILED;
    }
    if (*in1_iter < *in2_iter) {
      out_accum->push_back(*in1_iter);
      ++in1_iter;
      continue;
    } else if (*in1_iter == *in2_iter) {
      out_accum->push_back(*in1_iter);
      ++in1_iter;
      ++in2_iter;
    } else {
      out_accum->push_back(*in2_iter);
      ++in2_iter;
    }
  }
  if ((in1_iter != in1_accum.end()) || (in2_iter != in2_accum.end())) {
    MS_LOG(ERROR) << "last element of in1 and in2 must be equal";
    return Status::FAILED;
  }
  return Status::SUCCESS;
}

/*
 * example:
 * in1 = [8, 4]
 * in2 = [2, 16]
 * out = [2, 4, 4]
 */
Status UnifyShape(const std::vector<int32_t>& in1, const std::vector<int32_t>& in2, std::vector<int32_t>* out) {
  MS_EXCEPTION_IF_NULL(out);
  std::vector<int64_t> in1_accum;
  Status status = ShapeToAccumulateProduct(in1, &in1_accum);
  if (status != Status::SUCCESS) {
    return status;
  }
  std::vector<int64_t> in2_accum;
  status = ShapeToAccumulateProduct(in2, &in2_accum);
  if (status != Status::SUCCESS) {
    return status;
  }
  std::vector<int64_t> out_accum;
  status = UnifyAccumulateProduct(in1_accum, in2_accum, &out_accum);
  if (status != Status::SUCCESS) {
    return status;
  }
  status = AccumulateProductToShape(out_accum, out);
  if (status != Status::SUCCESS) {
    return status;
  }
  return status;
}

/*
 * example1:
 * in_accum_reverse = [2 * 8 * 32, 8 * 32, 32]
 * expand_accum_reverse = [2 * 8 * 32, 32, 8]
 * out_accum_reverse = [2 * 8 * 4 * 8, 8 * 4 * 8, 4 * 8, 8]
 *
 * example2:
 * in_accum_reverse = [2 * 8 * 32, 8 * 32, 32]
 * expand_accum_reverse = [2 * 4 * 8, 4 * 8, 8]
 * out_accum_reverse = [2 * 4 * 2 * 4 * 8, 4 * 2 * 4 * 8, 2 * 4 * 8, 4 * 8, 8]
 */
Status ExpandAccumulateProduct(const std::vector<int64_t>& in_accum_reverse,
                               const std::vector<int64_t>& expand_accum_reverse,
                               std::vector<int64_t>* out_accum_reverse) {
  MS_EXCEPTION_IF_NULL(out_accum_reverse);
  out_accum_reverse->clear();
  auto in_riter = in_accum_reverse.rbegin();
  auto expand_riter = expand_accum_reverse.rbegin();
  while (expand_riter != expand_accum_reverse.rend()) {
    if (in_riter == in_accum_reverse.rend()) {
      MS_LOG(ERROR) << "invalid ExpandAccumProd inputs";
      return Status::FAILED;
    }
    if (*in_riter > *expand_riter) {
      (void)out_accum_reverse->insert(out_accum_reverse->begin(), *expand_riter);
      ++expand_riter;
    } else if (*in_riter == *expand_riter) {
      (void)out_accum_reverse->insert(out_accum_reverse->begin(), *expand_riter);
      ++in_riter;
      ++expand_riter;
    } else {
      (void)out_accum_reverse->insert(out_accum_reverse->begin(), *in_riter);
      ++in_riter;
    }
  }
  while (in_riter != in_accum_reverse.rend()) {
    (void)out_accum_reverse->insert(out_accum_reverse->begin(), *in_riter);
    ++in_riter;
  }
  return Status::SUCCESS;
}

/*
 * example1:
 * in = [2, 8, 32]
 * expand = [16, 4, 8]
 * out = [2, 8, 4, 8]
 *
 * example2:
 * in = [2, 8, 32]
 * expand = [2, 4, 8]
 * out = [2, 4, 2, 4, 8]
 */
Status ExpandShape(const std::vector<int32_t>& in, const std::vector<int32_t>& expand, std::vector<int32_t>* out) {
  MS_EXCEPTION_IF_NULL(out);
  std::vector<int64_t> in_accum_reverse;
  Status status = ShapeToAccumulateProductReverse(in, &in_accum_reverse);
  if (status != Status::SUCCESS) {
    return status;
  }
  std::vector<int64_t> expand_accum_reverse;
  status = ShapeToAccumulateProductReverse(expand, &expand_accum_reverse);
  if (status != Status::SUCCESS) {
    return status;
  }
  std::vector<int64_t> out_accum_reverse;
  status = ExpandAccumulateProduct(in_accum_reverse, expand_accum_reverse, &out_accum_reverse);
  if (status != Status::SUCCESS) {
    return status;
  }
  status = AccumulateProductReverseToShape(out_accum_reverse, out);
  if (status != Status::SUCCESS) {
    return status;
  }
  return status;
}

}  // namespace parallel
}  // namespace mindspore
