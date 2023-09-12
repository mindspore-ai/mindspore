/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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

/*!
 * \file common_shape_fns.cpp
 * \brief
 */
#include "common_shape_fns.h"
#include <vector>
#include <limits>
#include <algorithm>
#include <utility>
#include <map>
#include "op_log.h"
#include "error_util.h"
#include "graph/utils/op_desc_utils.h"
#include "common/util/error_manager/error_manager.h"
#include "util.h"

namespace ge {
const std::map<std::string, DataType> dtype_maps{{"DT_FLOAT", DT_FLOAT},
                                                 {"DT_FLOAT16", DT_FLOAT16},
                                                 {"DT_INT8", DT_INT8},
                                                 {"DT_INT16", DT_INT16},
                                                 {"DT_UINT16", DT_UINT16},
                                                 {"DT_UINT8", DT_UINT8},
                                                 {"DT_INT32", DT_INT32},
                                                 {"DT_INT64", DT_INT64},
                                                 {"DT_UINT32", DT_UINT32},
                                                 {"DT_UINT64", DT_UINT64},
                                                 {"DT_BOOL", DT_BOOL},
                                                 {"DT_DOUBLE", DT_DOUBLE},
                                                 {"DT_STRING", DT_STRING},
                                                 {"DT_DUAL_SUB_INT8", DT_DUAL_SUB_INT8},
                                                 {"DT_DUAL_SUB_UINT8", DT_DUAL_SUB_UINT8},
                                                 {"DT_COMPLEX64", DT_COMPLEX64},
                                                 {"DT_COMPLEX128", DT_COMPLEX128},
                                                 {"DT_QINT8", DT_QINT8},
                                                 {"DT_QINT16", DT_QINT16},
                                                 {"DT_QINT32", DT_QINT32},
                                                 {"DT_QUINT8", DT_QUINT8},
                                                 {"DT_QUINT16", DT_QUINT16},
                                                 {"DT_RESOURCE", DT_RESOURCE},
                                                 {"DT_STRING_REF", DT_STRING_REF},
                                                 {"DT_DUAL", DT_DUAL},
                                                 {"DT_UNDEFINED", DT_UNDEFINED}};

graphStatus WithRankAtLeast(const TensorDesc &tensor, int64_t rank, Shape &out, const char *op_name) {
  if (rank > INT32_MAX) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), ConcatString("rank[", rank, "] cannot exceed kint32max"));
    return GRAPH_FAILED;
  }
  Shape s = tensor.GetShape();
  std::vector<int64_t> dims = s.GetDims();
  // dim.size() convert to be type int64_t can't overflow
  int64_t size = static_cast<int64_t>(dims.size());
  if (!((size >= rank) || (dims == UNKNOWN_SHAPE))) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name),
                                        ConcatString("rank[", size, "] must be at least [", rank, "]"));
    return GRAPH_FAILED;
  }
  out = s;
  return GRAPH_SUCCESS;
}

graphStatus WithRankAtLeast(const GeTensorDescPtr &tensorDesc, int64_t rank, GeShape &out_shape, const char *op_name) {
  if (rank > INT32_MAX) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), ConcatString("rank[", rank, "] cannot exceed kint32max"));
    return GRAPH_FAILED;
  }

  GeShape s = tensorDesc->GetShape();
  std::vector<int64_t> dims = s.GetDims();
  // dim.size() convert to be type int64_t can't overflow
  int64_t size = static_cast<int64_t>(dims.size());
  if ((dims != UNKNOWN_RANK) && (size < rank)) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name),
                                        ConcatString("rank[", size, "] must be at least [", rank, "]"));
    return GRAPH_FAILED;
  }
  out_shape = s;
  return GRAPH_SUCCESS;
}

graphStatus WithRankShape(GeShape &shape, int64_t rank, const char *op_name) {
  if (rank > INT32_MAX) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), ConcatString("rank[", rank, "] cannot exceed kint32max"));
    return GRAPH_FAILED;
  }

  int64_t existing = static_cast<int64_t>(shape.GetDimNum());

  if (shape.GetDims() == UNKNOWN_RANK) {
    std::vector<int64_t> out_shape(rank, UNKNOWN_DIM);
    shape = GeShape(out_shape);
    return GRAPH_SUCCESS;
  }
  if (existing != rank) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), ConcatString("rank[", existing, "] must be [", rank, "]"));
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dim_values = shape.GetDims();
  shape = GeShape(dim_values);
  return GRAPH_SUCCESS;
}

graphStatus WithRank(const TensorDesc &tensor, int64_t rank, Shape &out, const char *op_name) {
  if (rank > INT32_MAX) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), ConcatString("rank[", rank, "] cannot exceed kint32max"));
    return GRAPH_FAILED;
  }
  Shape s = tensor.GetShape();
  int64_t existing = static_cast<int64_t>(s.GetDimNum());

  if (s.GetDims() == UNKNOWN_RANK) {
    std::vector<int64_t> out_shape(rank, UNKNOWN_DIM);
    out = Shape(out_shape);
    return GRAPH_SUCCESS;
  }

  if (existing != rank) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), ConcatString("rank[", existing, "] must be [", rank, "]"));
    return GRAPH_FAILED;
  }
  out = s;
  return GRAPH_SUCCESS;
}

graphStatus WithRank(const GeTensorDescPtr &tensorDesc, int64_t rank, GeShape &out_shape, const char *op_name) {
  if (rank > INT32_MAX) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), ConcatString("rank[", rank, "] cannot exceed kint32max"));
    return GRAPH_FAILED;
  }

  GeShape s = tensorDesc->GetShape();
  int64_t existing = static_cast<int64_t>(s.GetDimNum());
  if (s.GetDims() == UNKNOWN_RANK) {
    std::vector<int64_t> out_dims(rank, UNKNOWN_DIM);
    out_shape = GeShape(out_dims);
    return GRAPH_SUCCESS;
  }

  if (existing != rank) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), ConcatString("rank[", existing, "] must be [", rank, "]"));
    return GRAPH_FAILED;
  }
  out_shape = s;
  return GRAPH_SUCCESS;
}

graphStatus WithRank(const GeTensorDescPtr &tensorDesc, int64_t rank, Shape &out_shape, const char *op_name) {
  if (rank > INT32_MAX) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), ConcatString("rank[", rank, "] cannot exceed kint32max"));
    return GRAPH_FAILED;
  }

  GeShape s = tensorDesc->GetShape();
  int64_t existing = static_cast<int64_t>(s.GetDimNum());
  if (s.GetDims() == UNKNOWN_RANK) {
    std::vector<int64_t> out_dims(rank, UNKNOWN_DIM);
    out_shape = Shape(out_dims);
    return GRAPH_SUCCESS;
  }

  if (existing != rank) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), ConcatString("rank[", existing, "] must be [", rank, "]"));
    return GRAPH_FAILED;
  }
  out_shape = Shape(s.GetDims());
  return GRAPH_SUCCESS;
}

graphStatus WithValue(int64_t dim, int64_t value, int64_t &out, const char *op_name) {
  out = value;
  if (dim == UNKNOWN_DIM) {
    return GRAPH_SUCCESS;
  }

  if (dim != value) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), ConcatString("dim[", dim, "] should be ", value));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

graphStatus MergePrefix(const Shape &s, const Shape &prefix, Shape &s_out, Shape &prefix_out, const char *op_name) {
  // Same shape and unknown rank
  if (!RankKnown(s) || !RankKnown(prefix)) {
    s_out = s;
    prefix_out = prefix;
    return GRAPH_SUCCESS;
  }
  const size_t rank = prefix.GetDimNum();
  std::vector<int64_t> dims1 = s.GetDims();
  if ((dims1 != UNKNOWN_RANK) && (dims1.size() < rank)) {
    std::string err_msg = ConcatString("first shape rank[", dims1.size(), "] must be at least rank[", rank, "]");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), err_msg);
    return GRAPH_FAILED;
  }

  const size_t rank_s = s.GetDimNum();
  std::vector<int64_t> dims;
  dims.reserve(std::max(rank, rank_s));
  dims.resize(rank);
  for (size_t i = 0; i < rank; ++i) {
    if (Merge(s.GetDim(i), prefix.GetDim(i), dims[i]) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString(i, "th dim of first shape", DebugString(s.GetDims()),
                                         " is not same as that of prefix shape", DebugString(prefix.GetDims()));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), err_msg);
      return GRAPH_FAILED;
    }
  }
  prefix_out = Shape(dims);
  for (size_t i = rank; i < rank_s; ++i) {
    dims.push_back(s.GetDim(i));
  }
  s_out = Shape(dims);
  return GRAPH_SUCCESS;
}

graphStatus Merge(int64_t dim1, int64_t dim2, int64_t &out) {
  if (dim1 == dim2) {
    out = dim1;
    return GRAPH_SUCCESS;
  } else if (dim2 == UNKNOWN_DIM) {
    out = dim1;
    return GRAPH_SUCCESS;
  } else if (dim1 == UNKNOWN_DIM) {
    out = dim2;
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

graphStatus Merge(const Shape &s0, const Shape &s1, Shape &out, const char *op_name) {
  // Same shape and unknown rank
  if (s0.GetDims() == s1.GetDims()) {
    out = s0;
    return GRAPH_SUCCESS;
  } else if (!RankKnown(s1)) {
    out = s0;
    return GRAPH_SUCCESS;
  } else if (!RankKnown(s0)) {
    out = s1;
    return GRAPH_SUCCESS;
  }

  const size_t rank = s0.GetDimNum();
  if (s1.GetDimNum() != rank) {
    std::string err_msg = ConcatString("different rank of first shape", DebugString(s0.GetDims()), " and second shape",
                                       DebugString(s1.GetDims()));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), err_msg);
    return GRAPH_FAILED;
  }

  // Check if each dims equal
  bool return_s0 = true;
  bool return_s1 = true;
  for (size_t i = 0; i < rank; i++) {
    int64_t d0 = s0.GetDim(i);
    int64_t d1 = s1.GetDim(i);
    if (d0 == UNKNOWN_DIM) {
      if (d1 != UNKNOWN_DIM) {
        return_s0 = false;
      }
    } else if (d1 == UNKNOWN_DIM) {
      return_s1 = false;
    } else if (d0 != d1) {
      std::string err_msg = ConcatString("different ", i, "th dim of first shape", DebugString(s0.GetDims()),
                                         " and second shape", DebugString(s1.GetDims()));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), err_msg);
      return GRAPH_FAILED;
    }
  }

  if (return_s0 || return_s1) {
    out = return_s0 ? s0 : s1;
    return GRAPH_SUCCESS;
  }

  // Merge dims
  std::vector<int64_t> dims(rank, 0);
  for (size_t i = 0; i < rank; ++i) {
    // Invariant for merge was checked earlier, so CHECK is ok.
    if (Merge(s0.GetDim(i), s1.GetDim(i), dims[i]) == GRAPH_FAILED) {
      std::string err_msg = ConcatString("merge ", i, "th dim failed, first shape", DebugString(s0.GetDims()),
                                         " and second shape", DebugString(s1.GetDims()));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), err_msg);
      return GRAPH_FAILED;
    }
  }

  out = Shape(dims);
  return GRAPH_SUCCESS;
}

graphStatus Merge(const GeShape &s0, const GeShape &s1, GeShape &out, const char *op_name) {
  // Same shape and unknown rank
  if (s0.GetDims() == s1.GetDims()) {
    out = s0;
    return GRAPH_SUCCESS;
  } else if (!RankKnown(s1)) {
    out = s0;
    return GRAPH_SUCCESS;
  } else if (!RankKnown(s0)) {
    out = s1;
    return GRAPH_SUCCESS;
  }

  const size_t rank = s0.GetDimNum();
  if (s1.GetDimNum() != rank) {
    std::string err_msg = ConcatString("different rank of first shape", DebugString(s0.GetDims()), " and second shape",
                                       DebugString(s1.GetDims()));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), err_msg);
    return GRAPH_FAILED;
  }

  // Check if each dims equal
  bool return_s0 = true;
  bool return_s1 = true;
  for (size_t i = 0; i < rank; i++) {
    int64_t d0 = s0.GetDim(i);
    int64_t d1 = s1.GetDim(i);
    if (d0 == UNKNOWN_DIM) {
      if (d1 != UNKNOWN_DIM) {
        return_s0 = false;
      }
    } else if (d1 == UNKNOWN_DIM) {
      return_s1 = false;
    } else if (d0 != d1) {
      std::string err_msg = ConcatString("different ", i, "th dim of first shape", DebugString(s0.GetDims()),
                                         " and second shape", DebugString(s1.GetDims()));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), err_msg);
      return GRAPH_FAILED;
    }
  }

  if (return_s0 || return_s1) {
    out = return_s0 ? s0 : s1;
    return GRAPH_SUCCESS;
  }

  // Merge dims
  std::vector<int64_t> dims(rank, 0);
  for (size_t i = 0; i < rank; ++i) {
    // Invariant for merge was checked earlier, so CHECK is ok.
    if (Merge(s0.GetDim(i), s1.GetDim(i), dims[i]) == GRAPH_FAILED) {
      std::string err_msg = ConcatString("merge ", i, "th dim failed, first shape", DebugString(s0.GetDims()),
                                         " and second shape", DebugString(s1.GetDims()));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), err_msg);
      return GRAPH_FAILED;
    }
  }

  out = GeShape(dims);
  return GRAPH_SUCCESS;
}

void MergeShape(const Shape &shared_shape, const Shape &value_shape, std::vector<int64_t> &out, bool &shape_changed) {
  for (size_t i = 0; i < out.size(); ++i) {
    if (shared_shape.GetDim(i) == value_shape.GetDim(i) || shared_shape.GetDim(i) == -1) {
      out[i] = shared_shape.GetDim(i);
    } else {
      out[i] = -1;
      shape_changed = true;
    }
  }
}

void MergeRange(const std::vector<std::pair<int64_t, int64_t>> &shared_shape_range,
                const std::vector<std::pair<int64_t, int64_t>> &value_shape_range,
                std::vector<std::pair<int64_t, int64_t>> &out, bool &shape_changed) {
  for (size_t i = 0; i < out.size(); ++i) {
    auto &shared_range = shared_shape_range[i];
    auto &value_range = value_shape_range[i];
    if (shared_range.first <= value_range.first) {
      out[i].first = shared_range.first;
    } else {
      out[i].first = value_range.first;
      shape_changed = true;
    }
    if (shared_range.second == -1 || (value_range.second != -1 && shared_range.second >= value_range.second)) {
      out[i].second = shared_range.second;
    } else {
      out[i].second = value_range.second;
      shape_changed = true;
    }
  }
}

graphStatus MergeShapeAndRange(const ShapeAndRange &shared_shape_and_range, const ShapeAndRange &value_shape_and_range,
                               ShapeAndRange &out, bool &shape_changed, const char *op_name) {
  if (!RankKnown(shared_shape_and_range.shape_)) {
    out = {Shape(UNKNOWN_RANK), {}, value_shape_and_range.shape_type_};
    return GRAPH_SUCCESS;
  }
  if (!RankKnown(value_shape_and_range.shape_) ||
      (shared_shape_and_range.shape_.GetDimNum() != value_shape_and_range.shape_.GetDimNum())) {
    out = {Shape(UNKNOWN_RANK), {}, value_shape_and_range.shape_type_};
    shape_changed = true;
    return GRAPH_SUCCESS;
  }
  auto actual_shared_range = shared_shape_and_range.shape_range_;
  auto actual_value_range = value_shape_and_range.shape_range_;
  if (shared_shape_and_range.shape_.GetDimNum() != shared_shape_and_range.shape_range_.size()) {
    actual_shared_range.clear();
    for (auto dim : shared_shape_and_range.shape_.GetDims()) {
      if (dim == ge::UNKNOWN_DIM) {
        actual_shared_range.push_back({1, -1});
      } else {
        actual_shared_range.push_back({dim, dim});
      }
    }
  }
  if (value_shape_and_range.shape_.GetDimNum() != value_shape_and_range.shape_range_.size()) {
    actual_value_range.clear();
    for (auto dim : value_shape_and_range.shape_.GetDims()) {
      if (dim == ge::UNKNOWN_DIM) {
        actual_value_range.push_back({1, -1});
      } else {
        actual_value_range.push_back({dim, dim});
      }
    }
  }
  const size_t rank = value_shape_and_range.shape_.GetDimNum();
  std::vector<int64_t> dims(rank);
  std::vector<std::pair<int64_t, int64_t>> shape_range(rank);
  MergeShape(shared_shape_and_range.shape_, value_shape_and_range.shape_, dims, shape_changed);
  MergeRange(actual_shared_range, actual_value_range, shape_range, shape_changed);
  out = {Shape(dims), shape_range, value_shape_and_range.shape_type_};
  return GRAPH_SUCCESS;
}

graphStatus ReplaceDim(const Shape &s, int64_t dim_index_in, int64_t new_dim, Shape &out, const char *op_name) {
  if (!RankKnown(s)) {
    out = Shape(ge::UNKNOWN_SHAPE);
    return GRAPH_SUCCESS;
  }
  int64_t dim_index = dim_index_in;
  if (dim_index < 0) {
    dim_index = (int64_t)s.GetDimNum() + dim_index;
  }
  if (!FastBoundsCheck(dim_index, s.GetDimNum())) {
    out = Shape();
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), ConcatString("out of range: replace dim[", dim_index_in,
                                                                      "] for shape with rank[", s.GetDimNum(), "]"));
    return GRAPH_FAILED;
  }
  std::vector<int64_t> dims = s.GetDims();
  dims[dim_index] = new_dim;
  out = Shape(dims);
  return GRAPH_SUCCESS;
}

graphStatus ReplaceDim(const GeShape &s, int64_t dim_index_in, int64_t new_dim, GeShape &out, const char *op_name) {
  if (!RankKnown(s)) {
    out = GeShape(UNKNOWN_RANK);
    return GRAPH_SUCCESS;
  }
  int64_t dim_index = dim_index_in;
  if (dim_index < 0) {
    dim_index = (int64_t)s.GetDimNum() + dim_index;
  }
  if (!FastBoundsCheck(dim_index, s.GetDimNum())) {
    out = GeShape();
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), ConcatString("out of range: replace dim[", dim_index_in,
                                                                      "] for shape with rank[", s.GetDimNum(), "]"));
    return GRAPH_FAILED;
  }
  std::vector<int64_t> dims = s.GetDims();
  dims[dim_index] = new_dim;
  out = GeShape(dims);
  return GRAPH_SUCCESS;
}

template <typename Ta, typename Tb>
bool FastBoundsCheck(const Ta index, const Tb limit) {
  static_assert(std::is_integral<Ta>::value && std::is_integral<Tb>::value,
                "FastBoundsCheck can only be used on integer types.");
  using UIndex = typename std::make_unsigned<decltype(index + limit)>::type;
  return static_cast<UIndex>(index) < static_cast<UIndex>(limit);
}

graphStatus Add(int64_t dim1, int64_t dim2, int64_t &out) {
  if (dim1 == 0) {
    out = dim2;
  } else if (dim2 == 0) {
    out = dim1;
  } else if ((dim1 == UNKNOWN_DIM) || (dim2 == UNKNOWN_DIM)) {
    out = UNKNOWN_DIM;
  } else {
    const int64_t sum = dim1 + dim2;
    if (sum < 0) {
      return GRAPH_FAILED;
    }
    out = sum;
  }
  return GRAPH_SUCCESS;
}

graphStatus Subtract(int64_t dim1, int64_t dim2, int64_t &out, const char *op_name) {
  if (dim2 == 0) {
    out = dim1;
  } else if ((dim1 == UNKNOWN_DIM) || (dim2 == UNKNOWN_DIM)) {
    out = UNKNOWN_DIM;
  } else {
    if (dim1 < dim2) {
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
        string(op_name), ConcatString("negative dimension caused by subtracting. dim1[", dim1, "], dim2[", dim2, "]"));
      return GRAPH_FAILED;
    }
    out = dim1 - dim2;
  }
  return GRAPH_SUCCESS;
}

graphStatus SubShape(const Shape &s, int64_t start, int64_t end, int64_t stride, Shape &out, const char *op_name) {
  if (s.GetDimNum() > INT32_MAX) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name),
                                        ConcatString("rank[", s.GetDimNum(), "] cannot exceed kint32max"));
    return GRAPH_FAILED;
  }
  const int64_t rank = static_cast<int64_t>(s.GetDimNum());
  TensorDesc tensor(s);
  if (!RankKnown(s) ||
      (start == 0 && ((tensor.GetRealDimCnt() != -1 && end >= rank) || end == std::numeric_limits<int64_t>::max()))) {
    out = s;
    return GRAPH_SUCCESS;
  }

  if (start > rank) {
    start = rank;
  }
  if (end > rank) {
    end = rank;
  }

  if (stride < 0 && start == rank) {
    --start;
  }

  if (start < 0) {
    start = rank + start;
    if (start < 0) {
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
        string(op_name), ConcatString("invalid start[", start - rank, "] to get sub shape with rank[", rank, "]"));
      return GRAPH_FAILED;
    }
  }

  if (end < 0) {
    end = rank + end;
    if (end < 0) {
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
        string(op_name), ConcatString("invalid end[", end - rank, "] to get sub shape with rank[", rank, "]"));
      return GRAPH_FAILED;
    }
  }

  // stride > 0 and start > end
  if (!((stride <= 0 || start <= end))) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), ConcatString("start[", start, "] should be less than end[",
                                                                      end, "] at positive stride[", stride, "]"));
    return GRAPH_FAILED;
  }
  // stride < 0 and start < end
  if (!(stride >= 0 || start >= end)) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), ConcatString("start[", start, "] should be greater than end[",
                                                                      end, "] at negative stride[", stride, "]"));
    return GRAPH_FAILED;
  }
  std::vector<int64_t> dims;
  for (int64_t i = start; stride > 0 ? i < end : i > end; i += stride) {
    dims.push_back(s.GetDim(i));
  }
  Shape tmp(dims);
  out = tmp;
  return GRAPH_SUCCESS;
}

graphStatus SubShape(const GeShape &src_shape, int64_t start, int64_t end, int64_t stride, GeShape &out_shape,
                     const char *op_name) {
  int64_t src_rank = src_shape.GetDimNum();
  if (src_rank > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), ConcatString("rank[", src_rank, "] cannot exceed kint32max"));
    return GRAPH_FAILED;
  }

  if (start == 0 && stride == 1 &&
      ((RankKnown(src_shape) && end >= src_rank) || (end == std::numeric_limits<int64_t>::max()))) {
    out_shape = src_shape;
    return GRAPH_SUCCESS;
  }

  if (start > src_rank) {
    start = src_rank;
  }

  if (end > src_rank) {
    end = src_rank;
  }

  if (stride < 0 && start == src_rank) {
    --start;
  }

  if (start < 0) {
    start += src_rank;
    if (start < 0) {
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
        string(op_name),
        ConcatString("invalid start[", start - src_rank, "] to get sub shape with rank[", src_rank, "]"));
      return GRAPH_FAILED;
    }
  }

  if (end < 0) {
    end += src_rank;
    if (end < 0) {
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
        string(op_name), ConcatString("invalid end[", end - src_rank, "] to get sub shape with rank[", src_rank, "]"));
      return GRAPH_FAILED;
    }
  }

  if (stride > 0 && start > end) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), ConcatString("start[", start, "] should be less than end[",
                                                                      end, "] at positive stride[", stride, "]"));
    return GRAPH_FAILED;
  } else if (stride < 0 && start < end) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), ConcatString("start[", start, "] should be greater than end[",
                                                                      end, "] at negative stride[", stride, "]"));
    return GRAPH_FAILED;
  }

  std::vector<int64_t> out_dims;
  for (int64_t i = start; (stride > 0 ? i < end : i > end); i += stride) {
    out_dims.push_back(src_shape.GetDim(i));
  }
  out_shape = GeShape(out_dims);
  return GRAPH_SUCCESS;
}

graphStatus Concatenate(const Shape &s1, const Shape &s2, Shape &out) {
  if (!RankKnown(s1) || !RankKnown(s2)) {
    out = Shape(ge::UNKNOWN_RANK);
    return GRAPH_SUCCESS;
  }
  size_t s1_rank = s1.GetDimNum();
  size_t s2_rank = s2.GetDimNum();
  size_t rank = s1_rank + s2_rank;
  std::vector<int64_t> dims;
  dims.reserve(rank);
  for (size_t i = 0; i < s1_rank; ++i) {
    dims.push_back(s1.GetDim(i));
  }
  for (size_t i = 0; i < s2_rank; ++i) {
    dims.push_back(s2.GetDim(i));
  }
  Shape s(dims);
  out = s;
  return GRAPH_SUCCESS;
}

graphStatus Concatenate(const GeShape &s1, const GeShape &s2, GeShape &out) {
  if (!RankKnown(s1) || !RankKnown(s2)) {
    out = GeShape(ge::UNKNOWN_RANK);
    return GRAPH_SUCCESS;
  }
  const int64_t s1_rank = s1.GetDimNum();
  const int64_t s2_rank = s2.GetDimNum();
  const int64_t out_rank = s1_rank + s2_rank;
  std::vector<int64_t> out_dims;
  out_dims.reserve(out_rank);
  for (int64_t i = 0; i < s1_rank; ++i) {
    out_dims.push_back(s1.GetDim(i));
  }
  for (int64_t i = 0; i < s2_rank; ++i) {
    out_dims.push_back(s2.GetDim(i));
  }
  out = GeShape(out_dims);
  return GRAPH_SUCCESS;
}

graphStatus Matrix(int64_t dim1, int64_t dim2, Shape &out) {
  std::vector<int64_t> dims;
  dims.reserve(DIM_VALUE2);  // The number of dims is 2.
  dims.push_back(dim1);
  dims.push_back(dim2);
  Shape s(dims);
  out = s;
  return GRAPH_SUCCESS;
}

graphStatus Vector(int64_t dim, Shape &out) {
  std::vector<int64_t> dims;
  dims.reserve(1);
  dims.push_back(dim);
  Shape s(dims);
  out = s;
  return GRAPH_SUCCESS;
}

static graphStatus GetShapeDataFromShapeTensor(Operator &op, const string &dst_name, int64_t rank,
                                               std::vector<int64_t> &data, const char *op_name) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto shape_data_desc = op_desc->MutableInputDesc(dst_name);

  std::vector<std::string> input_infer_depends = {dst_name};
  op_desc->SetOpInferDepends(input_infer_depends);

  GeShape shape_data_shape(shape_data_desc->GetShape());
  std::vector<int64_t> dims = shape_data_shape.GetDims();
  DataType data_type = shape_data_desc->GetDataType();
  if (dims.size() != static_cast<size_t>(rank)) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
      string(op_name), ConcatString("invalid shape data rank[", dims.size(), "], should be [", rank, "]"));
    return GRAPH_FAILED;
  }
  int64_t dim_value = ((rank > 0) && (dims[0] > 0)) ? dims[0] : 1;
  data.clear();
  if (dims[0] < 0) {
    OP_LOGI(op_name, "Shape rank is %zu, dims[0] value is [%ld]", dims.size(), dims[0]);
    data.push_back(UNKNOWN_DIM_NUM);
    return GRAPH_SUCCESS;
  }
  data.reserve(dim_value);
  Tensor shape_tensor;
  if (data_type == DT_INT32) {
    if (op.GetInputConstData(dst_name.c_str(), shape_tensor) == GRAPH_SUCCESS) {
      const int32_t *shape_data = reinterpret_cast<const int32_t *>(shape_tensor.GetData());
      for (int64_t i = 0; i < dim_value; i++) {
        data.push_back(static_cast<int64_t>(shape_data[i]));
      }
    } else {
      OP_LOGI(TbeGetName(op).c_str(), "Input [%s] is not a const tensor.", dst_name.c_str());
      for (int64_t i = 0; i < dim_value; i++) {
        data.push_back(UNKNOWN_DIM);
      }
    }
  } else if (data_type == DT_INT64) {
    if (op.GetInputConstData(dst_name.c_str(), shape_tensor) == GRAPH_SUCCESS) {
      const int64_t *shape_data = reinterpret_cast<const int64_t *>(shape_tensor.GetData());
      for (int64_t i = 0; i < dim_value; i++) {
        data.push_back(static_cast<int64_t>(shape_data[i]));
      }
    } else {
      OP_LOGI(TbeGetName(op).c_str(), "Input [%s] is not a const tensor.", dst_name.c_str());
      for (int64_t i = 0; i < dim_value; i++) {
        data.push_back(UNKNOWN_DIM);
      }
    }
  } else {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
      string(op_name), ConcatString("invalid data type[", DTypeStr(data_type), "], should be DT_INT32 or DT_INT64"));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

static graphStatus GetShapeDataFromConstData(const Tensor &tensor, int64_t rank, std::vector<int64_t> &data,
                                             const char *op_name) {
  TensorDesc shape_data_desc = tensor.GetTensorDesc();
  Shape shape_data_shape = shape_data_desc.GetShape();
  std::vector<int64_t> dims = shape_data_shape.GetDims();
  DataType data_type = shape_data_desc.GetDataType();

  if (dims.size() != static_cast<size_t>(rank)) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
      string(op_name), ConcatString("invalid shape data rank[", dims.size(), "], should be [", rank, "]"));
    return GRAPH_FAILED;
  }
  int64_t dim_value = rank > 0 ? dims[0] : 1;
  OP_LOGI(op_name, "data_type = %d, dim_value = %ld", data_type, dim_value);
  data.clear();
  data.reserve(dim_value);
  if (data_type == DT_INT32) {
    const int32_t *shape_data = reinterpret_cast<const int32_t *>(tensor.GetData());
    for (int64_t i = 0; i < dim_value; i++) {
      OP_LOGI(op_name, "DT_INT32 i = %ld, shape_data[i] = %ld", i, static_cast<int64_t>(shape_data[i]));
      data.push_back(static_cast<int64_t>(shape_data[i]));
    }
  } else if (data_type == DT_INT64) {
    const int64_t *shape_data = reinterpret_cast<const int64_t *>(tensor.GetData());
    for (int64_t i = 0; i < dim_value; i++) {
      OP_LOGI(op_name, "DT_INT64 i = %ld, shape_data[i] = %ld", i, shape_data[i]);
      data.push_back(shape_data[i]);
    }
  } else {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
      string(op_name), ConcatString("invalid data type[", DTypeStr(data_type), "], should be DT_INT32 or DT_INT64"));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

graphStatus MakeShapeFromShapeTensor(const Tensor &tensor, Shape &out, const char *op_name) {
  std::vector<int64_t> shape_data;
  GetShapeDataFromConstData(tensor, 1, shape_data, op_name);
  out = Shape(shape_data);
  return GRAPH_SUCCESS;
}

graphStatus MakeShapeFromShapeTensor(Operator &op, const string &dst_name, GeShape &out, const char *op_name) {
  std::vector<int64_t> shape_data;
  if (GetShapeDataFromShapeTensor(op, dst_name, 1, shape_data, op_name) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  out = GeShape(shape_data);
  return GRAPH_SUCCESS;
}

graphStatus MakeDimForScalarInput(const Tensor &tensor, int64_t &out, const char *op_name) {
  std::vector<int64_t> shape_data;
  GetShapeDataFromConstData(tensor, 0, shape_data, op_name);
  out = shape_data[0];
  return GRAPH_SUCCESS;
}

graphStatus WithRankAtMost(const TensorDesc &tensor, int64_t rank, Shape &out, const char *op_name) {
  if (rank > INT32_MAX) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), ConcatString("rank[", rank, "] cannot exceed kint32max"));
    return GRAPH_FAILED;
  }
  Shape s = tensor.GetShape();
  std::vector<int64_t> dims = s.GetDims();
  if (!((dims.size() <= static_cast<size_t>(rank)) || (dims == ge::UNKNOWN_SHAPE))) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name),
                                        ConcatString("invalid rank[", dims.size(), "], should be at most ", rank));
    return GRAPH_FAILED;
  }
  out = s;
  return GRAPH_SUCCESS;
}

graphStatus WithRankAtMost(const GeTensorDescPtr &tensorDesc, int64_t rank, GeShape &out_shape, const char *op_name) {
  if (rank > INT32_MAX) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name), ConcatString("rank[", rank, "] cannot exceed kint32max"));
    return GRAPH_FAILED;
  }

  GeShape s = tensorDesc->GetShape();
  std::vector<int64_t> dims = s.GetDims();
  if ((dims != ge::UNKNOWN_RANK) && (dims.size() > static_cast<size_t>(rank))) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name),
                                        ConcatString("invalid rank[", dims.size(), "], should be at most ", rank));
    return GRAPH_FAILED;
  }

  out_shape = s;
  return GRAPH_SUCCESS;
}

graphStatus Scalar(Shape &out) {
  std::vector<int64_t> dims = {};
  Shape s(dims);
  out = s;
  return GRAPH_SUCCESS;
}

graphStatus UnchangedShape(Operator &op, const string input_name, const string output_name) {
  TensorDesc desc = op.GetOutputDescByName(output_name.c_str());
  desc.SetShape(op.GetInputDescByName(input_name.c_str()).GetShape());
  return op.UpdateOutputDesc(output_name.c_str(), desc);
}

graphStatus Divide(const int64_t dividend, const int64_t divisor, const bool evenlyDivisible, int64_t &out,
                   const char *op_name) {
  if (divisor == 1) {
    out = dividend;
  } else if ((dividend == ge::UNKNOWN_DIM) || (divisor == ge::UNKNOWN_DIM)) {
    out = ge::UNKNOWN_DIM;
  } else {
    if (divisor <= 0) {
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name),
                                          ConcatString("invalid divisor[", divisor, "], should be positive"));
      return GRAPH_FAILED;
    }
    if (!((!evenlyDivisible) || (dividend % divisor) == 0)) {
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
        string(op_name), ConcatString("[", dividend, "] cannot be evenly divisible by [", divisor, "]"));
      return GRAPH_FAILED;
    }
    out = dividend / divisor;
  }
  return GRAPH_SUCCESS;
}

bool ShapeFullDefined(const Shape &shape) {
  if (!RankKnown(shape)) {
    return false;
  }
  std::vector<int64_t> dims = shape.GetDims();

  for (const auto &dim : dims) {
    if (dim == ge::UNKNOWN_DIM) {
      return false;
    }
  }
  return true;
}

bool ShapeFullyDefined(const GeShape &shape) {
  if (!RankKnown(shape)) {
    return false;
  }

  std::vector<int64_t> dims = shape.GetDims();
  for (const int64_t &dim : dims) {
    if (dim == ge::UNKNOWN_DIM) {
      return false;
    }
  }

  return true;
}

bool RankKnown(const Shape &shape) {
  std::vector<int64_t> dims = shape.GetDims();
  if (dims == ge::UNKNOWN_RANK) {
    return false;
  }
  return true;
}

bool RankKnown(const GeShape &shape) {
  std::vector<int64_t> dims = shape.GetDims();
  if (dims == ge::UNKNOWN_RANK) {
    return false;
  }
  return true;
}

Shape UnknownShapeOfRank(int64_t rank) {
  std::vector<int64_t> dims(rank);
  for (int64_t i = 0; i < rank; ++i) {
    dims[i] = ge::UNKNOWN_DIM;
  }
  return Shape(dims);
}

bool ValueKnown(const Shape &shape, const size_t &dim_index) {
  if (shape.GetDims() == ge::UNKNOWN_SHAPE) {
    return false;
  }
  if (dim_index >= shape.GetDims().size()) {
    return false;
  }
  if (shape.GetDim(dim_index) == ge::UNKNOWN_DIM) {
    return false;
  }

  return true;
}

graphStatus ValidateSparseTensor(const TensorDesc &indices, const TensorDesc &values, const TensorDesc &shape,
                                 const char *op_name) {
  // Validate ranks
  Shape unused_shape;
  if (WithRank(indices, NUM_VALUE2, unused_shape, op_name) != GRAPH_SUCCESS) {  // The rank is 2.
    std::string err_msg = ConcatString("failed to call WithRank function, indices has wrong shape",
                                       DebugString(indices.GetShape().GetDims()), ", it should be 2D");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(string(op_name), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(values, 1, unused_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call WithRank function, values has wrong shape",
                                       DebugString(values.GetShape().GetDims()), ", it should be 1D");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(string(op_name), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(shape, 1, unused_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call WithRank function, shape has wrong shape",
                                       DebugString(shape.GetShape().GetDims()), ", it should be 1D");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(string(op_name), err_msg);
    return GRAPH_FAILED;
  }

  // Number of elements in indices and values must match
  Shape indices_shape = indices.GetShape();
  Shape values_shape = values.GetShape();
  if (ValueKnown(indices_shape, 0)) {
    if (ValueKnown(values_shape, 0)) {
      if (indices_shape.GetDim(0) != values_shape.GetDim(0)) {
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
          string(op_name), ConcatString("dim[0] of indices and dim[0] of value do not match, ", indices_shape.GetDim(0),
                                        " and ", values_shape.GetDim(0)));
        return GRAPH_FAILED;
      }
    }
  }

  // Rank embedded in indices must match shape.
  Shape sparse_shape = shape.GetShape();
  if (ValueKnown(indices_shape, 1)) {
    if (ValueKnown(sparse_shape, 0)) {
      if (indices_shape.GetDim(1) != sparse_shape.GetDim(0)) {
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(string(op_name),
                                            ConcatString("dim[1] of indices and dim[0] of sparse do not match, ",
                                                         indices_shape.GetDim(1), " and ", sparse_shape.GetDim(0)));
        return GRAPH_FAILED;
      }
    }
  }
  return GRAPH_SUCCESS;
}

void FillOpDesc(GeTensorDescPtr &op_desc, const GeShape &shape, const DataType &data_type) {
  if (RankKnown(shape)) {
    auto dims = shape.GetDims();
    bool shape_fully_defined = true;
    for (const int64_t &dim : dims) {
      if (dim == UNKNOWN_DIM) {
        shape_fully_defined = false;
        break;
      }
    }
    if (!shape_fully_defined) {
      std::vector<std::pair<int64_t, int64_t>> shape_range;
      for (const int64_t &dim : dims) {
        shape_range.push_back(dim == UNKNOWN_DIM ? std::pair<int64_t, int64_t>{1, -1}
                                                 : std::pair<int64_t, int64_t>{dim, dim});
      }
      op_desc->SetShapeRange(shape_range);
    }
  }
  op_desc->SetShape(shape);
  op_desc->SetDataType(data_type);
}

void FillOpDesc(TensorDesc &op_desc, const Shape &shape, const DataType &data_type) {
  if (RankKnown(shape)) {
    auto dims = shape.GetDims();
    bool shape_fully_defined = true;
    for (const int64_t &dim : dims) {
      if (dim == UNKNOWN_DIM) {
        shape_fully_defined = false;
        break;
      }
    }
    if (!shape_fully_defined) {
      std::vector<std::pair<int64_t, int64_t>> shape_range;
      for (const int64_t &dim : dims) {
        shape_range.push_back(dim == UNKNOWN_DIM ? std::pair<int64_t, int64_t>{1, -1}
                                                 : std::pair<int64_t, int64_t>{dim, dim});
      }
      op_desc.SetShapeRange(shape_range);
    }
  }
  op_desc.SetShape(shape);
  op_desc.SetDataType(data_type);
}

void InferShapeErrorReport(const std::string &op_name, const std::string &op_type, const std::string &value,
                           const std::string &reason) {
  std::string report_error_code = "E14001";
  ErrorManager::GetInstance().ATCReportErrMessage(report_error_code, {"opname", "optype", "value", "reason"},
                                                  {op_name, op_type, value, reason});
}

std::string DTypeStr(DataType dtype) {
  auto iter =
    std::find_if(dtype_maps.begin(), dtype_maps.end(),
                 [dtype](const std::map<std::string, DataType>::value_type &kv) { return (kv.second == dtype); });
  if (iter != dtype_maps.end()) {
    return iter->first;
  } else {
    return std::string("DT_UNDEFINED");
  }
}

graphStatus SetShapeAndRange(Operator &op, const ShapeAndRange &feed_shape_and_range) {
  AscendString op_name;
  op.GetName(op_name);
  auto context = op.GetInferenceContext();
  std::vector<AscendString> marks;
  context->GetMarks(marks);

  if (!marks.empty()) {
    OP_LOGI(TbeGetName(op).c_str(), "Set marks[0] = %s", marks[0].GetString());
    bool shape_changed = false;
    auto aicpu_resource_context = reinterpret_cast<AicpuResourceContext *>(context->GetResourceContext(marks[0]));
    if (aicpu_resource_context == nullptr) {
      aicpu_resource_context = new (std::nothrow) AicpuResourceContext();
      if (aicpu_resource_context == nullptr) {
        AICPU_INFER_SHAPE_INNER_ERR_REPORT(std::string(op_name.GetString()),
                                           std::string("new AicpuResourceContext failed."));
        return GRAPH_FAILED;
      }
      aicpu_resource_context->shape_and_range_.push_back(feed_shape_and_range);
      if (context->SetResourceContext(marks[0], aicpu_resource_context) != GRAPH_SUCCESS) {
        delete aicpu_resource_context;
        AICPU_INFER_SHAPE_CALL_ERR_REPORT(std::string(op_name.GetString()),
                                          std::string("set resource context failed."));
        return GRAPH_FAILED;
      }
      shape_changed = true;
    } else {
      auto &shape_and_range = aicpu_resource_context->shape_and_range_;
      if (shape_and_range.empty()) {
        AICPU_INFER_SHAPE_CALL_ERR_REPORT(std::string(op_name.GetString()),
                                          std::string("get resource context shape and ranges failed."));
        return GRAPH_FAILED;
      }
      MergeShapeAndRange(shape_and_range[0], feed_shape_and_range, shape_and_range[0], shape_changed,
                         op_name.GetString());
    }
    if (shape_changed) {
      if (context->AddChangedResourceKey(marks[0]) != GRAPH_SUCCESS) {
        AICPU_INFER_SHAPE_CALL_ERR_REPORT(std::string(op_name.GetString()),
                                          std::string("add change resource key failed."));
        return GRAPH_FAILED;
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus GetShapeAndRange(Operator &op, ShapeAndRange &out, bool &geted, InferenceContextPtr infer_context) {
  AscendString op_name;
  op.GetName(op_name);
  std::vector<AscendString> marks;
  infer_context->GetMarks(marks);
  if (!marks.empty()) {
    OP_LOGI(TbeGetName(op).c_str(), "Get marks[0] = %s", marks[0].GetString());
    if (infer_context->RegisterReliedOnResourceKey(marks[0]) != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(std::string(op_name.GetString()),
                                         std::string("register relied on resource key failed."));
      return GRAPH_FAILED;
    }
    auto aicpu_resource_context = reinterpret_cast<AicpuResourceContext *>(infer_context->GetResourceContext(marks[0]));
    if (aicpu_resource_context != nullptr) {
      auto &shape_and_range = aicpu_resource_context->shape_and_range_;
      if (shape_and_range.empty()) {
        AICPU_INFER_SHAPE_INNER_ERR_REPORT(std::string(op_name.GetString()),
                                           std::string("get resource context shape and ranges failed."));
        return GRAPH_FAILED;
      }
      out.shape_ = shape_and_range[0].shape_;
      out.shape_range_ = shape_and_range[0].shape_range_;
      out.shape_type_ = shape_and_range[0].shape_type_;
      geted = true;
    }
  }
  return GRAPH_SUCCESS;
}
}  // namespace ge
