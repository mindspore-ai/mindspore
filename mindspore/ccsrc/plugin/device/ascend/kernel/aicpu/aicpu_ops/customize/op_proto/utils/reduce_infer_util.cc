/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
 * \file reduce_infer_util.cc
 * \brief
 */
#include <string>
#include <vector>
#include "util.h"
#include "reduce_infer_util.h"
#include "vector_proto_profiling.h"

namespace reduce_ops {
using namespace std;
using namespace ge;

constexpr int64_t UNKNOWN_DIM_VALUE = -2;

static bool ConvertAxis(std::vector<int64_t> &axis, const int64_t input_len) {
  const int64_t input_length = input_len == 0 ? 1 : input_len;
  // Convert reduce axis
  for (size_t i = 0; i < axis.size(); ++i) {
    if (axis[i] < -input_length || axis[i] > (input_length - 1)) {
      OP_LOGE("ReduceOps", "reduce verify failed, axis: %ld, input_length: %ld", axis[i], input_length);
      return false;
    }
    if (axis[i] < 0) {
      axis[i] = input_length + axis[i];
    }
  }
  return true;
}

bool DoReduceInfershapeWithAxesKeepdims(const GeShape &input_shape, std::vector<int64_t> &reduce_axes,
                                        GeShape &output_shape) {
  // case0: input is {-2}， set the output {-2}
  if (input_shape.IsUnknownDimNum()) {
    OP_LOGD("ReduceOps", "do unknownrank infershape for Reduce, output is {-2}");
    output_shape.SetIsUnknownDimNum();
    return true;
  }

  auto input_shape_len = input_shape.GetDimNum();
  OP_LOGD("ReduceOps", "input shape = %s, axes = %s", to_string(input_shape).c_str(), to_string(reduce_axes).c_str());
  if (!ConvertAxis(reduce_axes, static_cast<int64_t>(input_shape_len))) {
    OP_LOGE("ReduceOps", "do ConvertAxis failed, will return false");
    return false;
  }

  // case1: will reduce all shape, if reduce_axes is empty
  if (reduce_axes.empty()) {
    // return the shape(all 1) when reduce_axes is empty and keep_dims = true
    OP_LOGD("ReduceOps", "do all reduce infershape for Reduce, output is {}");
    output_shape.SetDimNum(input_shape_len);
    for (size_t i = 0; i < input_shape_len; ++i) {
      output_shape.SetDim(i, 1);
    }
    return true;
  }

  // case2: shape is [x, y, z] axis is [0] --> [1, y, z] when keep_dims is true
  output_shape.SetDimNum(input_shape_len);
  OP_LOGD("ReduceOps", "do norm infershape for Reduce");
  output_shape = input_shape;
  for (size_t i = 0; i < reduce_axes.size(); ++i) {
    int64_t axis = reduce_axes[i];
    output_shape.SetDim(axis, 1);
  }
  OP_LOGD("ReduceOps", "after reduce output shape = %s", to_string(output_shape).c_str());
  return true;
}

bool DoReduceInfershapeWithAxesNoKeepdims(const GeShape &input_shape, std::vector<int64_t> &reduce_axes,
                                          GeShape &output_shape) {
  // case0: will reduce all shape, if reduce_axes is empty
  if (reduce_axes.empty()) {
    // return a scalar shape when reduce_axes is empty and keep_dims = false
    OP_LOGD("ReduceOps", "reduce_axes is empty, output a scalar");
    output_shape.SetDimNum(0);
    return true;
  }

  // case1: input is {-2}， set the output {-2}
  if (input_shape.IsUnknownDimNum()) {
    OP_LOGD("ReduceOps", "input is {-2}, set the output is {-2}");
    output_shape.SetIsUnknownDimNum();
    return true;
  }

  auto input_shape_len = input_shape.GetDimNum();
  if (!ConvertAxis(reduce_axes, static_cast<int64_t>(input_shape_len))) {
    OP_LOGE("ReduceOps", "do ConvertAxis failed, will return false");
    return false;
  }
  // case2: shape is [x, y, z] axis is [0] --> [y, z] when keep_dims is false
  output_shape.SetDimNum(input_shape_len);
  int64_t output_dim = 0;
  for (int64_t i = 0; i < static_cast<int64_t>(input_shape_len); ++i) {
    if (std::find(reduce_axes.begin(), reduce_axes.end(), i) == reduce_axes.end()) {
      auto input_dim = input_shape.GetDim(i);
      output_shape.SetDim(output_dim, input_dim);
      output_dim++;
    }
  }
  output_shape.SetDimNum(output_dim);

  OP_LOGD("ReduceOps", "after reduce output shape = %s", to_string(output_shape).c_str());
  return true;
}

bool DoReduceInfershapeWithAxes(const GeShape &input_shape, const bool keep_dims, std::vector<int64_t> &reduce_axes,
                                GeShape &output_shape) {
  if (keep_dims) {
    return DoReduceInfershapeWithAxesKeepdims(input_shape, reduce_axes, output_shape);
  }

  return DoReduceInfershapeWithAxesNoKeepdims(input_shape, reduce_axes, output_shape);
}

bool DoReduceInferRangeWithAxes(GeTensorDescPtr &tensordesc_input_x, GeTensorDescPtr &tensordesc_output,
                                std::vector<int64_t> &reduce_axes, bool keep_dims) {
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  std::vector<std::pair<int64_t, int64_t>> input_shape_range;
  tensordesc_input_x->GetShapeRange(input_shape_range);
  std::vector<int64_t> input_shape_vec = tensordesc_input_x->MutableShape().GetDims();
  // If InputShapeRange is None, MakeUpShapeRange will set range.
  MakeUpShapeRange(input_shape_vec, input_shape_range);
  if (keep_dims) {
    output_shape_range = input_shape_range;
    for (auto item : reduce_axes) {
      output_shape_range[item] = std::make_pair<int64_t, int64_t>(1, 1);
    }
  } else {
    for (int64_t i = 0; i < static_cast<int64_t>(input_shape_range.size()); ++i) {
      if (std::find(reduce_axes.begin(), reduce_axes.end(), i) == reduce_axes.end()) {
        output_shape_range.push_back(input_shape_range[i]);
      }
    }
  }
  tensordesc_output->SetShapeRange(output_shape_range);
  return true;
}

bool GetConstData(const Operator &op, const int64_t const_input_idx, std::vector<int64_t> &const_values) {
  ge::Tensor const_tensor;
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto input_name = op_desc->GetInputNameByIndex(const_input_idx);
  if (op.GetInputConstData(input_name.c_str(), const_tensor) != ge::GRAPH_SUCCESS) {
    OP_LOGW(TbeGetName(op).c_str(), "constvalue [%s] not exists.", input_name.c_str());
    return false;
  }

  DataType const_dtype = op_desc->MutableInputDesc(const_input_idx)->GetDataType();
  auto size = const_tensor.GetSize();
  auto data = const_tensor.GetData();
  const_values.reserve(size);
  switch (const_dtype) {
    case ge::DT_INT64: {
      size_t count = size / sizeof(int64_t);
      const int64_t *data_addr = (const int64_t *)(data);
      for (size_t i = 0; i < count; i++) {
        const_values.push_back(*(data_addr + i));
      }
    } break;
    case ge::DT_INT32: {
      size_t count = size / sizeof(int32_t);
      const int32_t *data_addr = (const int32_t *)(data);
      for (size_t i = 0; i < count; i++) {
        const_values.push_back(*(data_addr + i));
      }
    } break;
    default: {
      OP_LOGW(TbeGetName(op).c_str(), "GetConstData of dtype[%s] has not implement.", to_string(const_dtype).c_str());
      return false;
    } break;
  }

  OP_LOGD(TbeGetName(op).c_str(), "get const value = %s", to_string(const_values).c_str());
  return true;
}

bool DoReduceInferShapeWithoutAxes(const Operator &op, GeTensorDescPtr &tensordesc_input_x,
                                   GeTensorDescPtr &tensordesc_output, const GeShape &axes_shape, bool keep_dims) {
  OP_LOGD(TbeGetName(op).c_str(), "the axes is not const, the output will be dynamic shape");
  const GeShape &input_shape = tensordesc_input_x->MutableShape();
  // case0: input is {}， set the output {}
  if (input_shape.IsScalar()) {
    OP_LOGD(TbeGetName(op).c_str(), "input is scalar, so output is scalar");
    std::vector<int64_t> output_shape;
    tensordesc_output->SetShape(GeShape(output_shape));
    return true;
  }
  // case1: input is {-2}， set the output {-2}
  if (input_shape.IsUnknownDimNum()) {
    OP_LOGD(TbeGetName(op).c_str(), "input is {-2}, so output {-2}");
    std::vector<int64_t> output_shape(1, -2);
    tensordesc_output->SetShape(GeShape(output_shape));
    return true;
  }

  std::vector<int64_t> input_shape_vec = input_shape.GetDims();
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  std::vector<std::pair<int64_t, int64_t>> input_shape_range;
  tensordesc_input_x->GetShapeRange(input_shape_range);
  // If InputShapeRange is None, MakeUpShapeRange will set range.
  MakeUpShapeRange(input_shape_vec, input_shape_range);
  size_t input_length = input_shape_vec.size();
  if (keep_dims) {
    // case2: all output shape dim is -1, range [1, xxx] when keep_dims is true
    for (size_t item = 0; item < input_length; ++item) {
      int64_t range_min_value = 1;
      int64_t range_max_value = input_shape_range[item].second;
      output_shape_range.push_back(std::make_pair(range_min_value, range_max_value));
      if (range_max_value == 1) {
        output_shape.push_back(1);
      } else {
        output_shape.push_back(-1);
      }
    }
  } else {
    // keep_dims is false
    // case3: all output shape dim is -1, range is (min_range, max_range)
    //            output dim num = (input dim num  - 1), when axes_shape = {} or {1}
    // case4: all output shape dim is -2
    int64_t output_dim_num = UNKNOWN_DIM_VALUE;
    if (!axes_shape.IsUnknownDimNum() && axes_shape.GetDimNum() == 0) {
      OP_LOGD(TbeGetName(op).c_str(), "the axes is scalar, will reduce one dim for input shape");
      output_dim_num = input_length - 1;
    }
    if (axes_shape.GetDimNum() == 1 && axes_shape.GetDim(0) == 1) {
      output_dim_num = input_length - 1;
      OP_LOGD(TbeGetName(op).c_str(), "the shape of axes is [1], will reduce one dim for input shape");
    }
    int64_t range_min_value = input_shape_range[0].first;
    int64_t range_max_value = input_shape_range[0].second;
    for (size_t item = 0; item < input_length; ++item) {
      if (input_shape_range[item].first < range_min_value) {
        range_min_value = input_shape_range[item].first;
      }
      if (input_shape_range[item].second == -1) {
        range_max_value = -1;
      }
      if (range_max_value != -1 && input_shape_range[item].second > range_max_value) {
        range_max_value = input_shape_range[item].second;
      }
    }
    if (output_dim_num == UNKNOWN_DIM_VALUE) {
      output_shape.push_back(UNKNOWN_DIM_VALUE);
    } else {
      for (int64_t item = 0; item < output_dim_num; ++item) {
        output_shape.push_back(-1);
        output_shape_range.push_back(std::make_pair(range_min_value, range_max_value));
      }
    }
  }
  tensordesc_output->SetShape(GeShape(output_shape));
  tensordesc_output->SetShapeRange(output_shape_range);
  return true;
}

bool CommonReduceInferWithInputAxes(const Operator &op, const int64_t input_x_idx, const int64_t output_idx,
                                    const int64_t input_axes_idx, bool keep_dims) {
  PROFILING_PROTO_INIT(TbeGetName(op).c_str());
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto axes_name = op_desc->GetInputNameByIndex(input_axes_idx);
  op_desc->SetOpInferDepends({axes_name});
  CHECK(op_desc == nullptr, VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("invalid OpDesc.")),
        return false);
  auto tensordesc_input_x = op_desc->MutableInputDesc(input_x_idx);
  auto tensordesc_input_axes = op_desc->MutableInputDesc(input_axes_idx);
  auto tensordesc_output = op_desc->MutableOutputDesc(output_idx);
  CHECK(tensordesc_input_x == nullptr || tensordesc_output == nullptr || tensordesc_input_axes == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("get mutable desc failed.")), return false);
  auto input_type = tensordesc_input_x->GetDataType();
  const GeShape &input_shape = tensordesc_input_x->MutableShape();
  const GeShape &axes_shape = tensordesc_input_axes->MutableShape();
  tensordesc_output->SetDataType(input_type);

  if (axes_shape.GetDimNum() == 1 && axes_shape.GetDim(0) == 0) {
    OP_LOGD(TbeGetName(op).c_str(), "axes_shape is [0], set output shape = input shape");
    tensordesc_output->SetShape(input_shape);
    std::vector<std::pair<int64_t, int64_t>> input_shape_range;
    tensordesc_input_x->GetShapeRange(input_shape_range);
    tensordesc_output->SetShapeRange(input_shape_range);
    return true;
  }

  // get const value from input_axes_idx
  std::vector<int64_t> reduce_axes;
  if (GetConstData(op, input_axes_idx, reduce_axes)) {
    PROFILING_PROTO_AFTER_GET_SHAPE_REG();
    // do infershape with const axes for static op
    GeShape &output_shape = tensordesc_output->MutableShape();
    CHECK(!DoReduceInfershapeWithAxes(input_shape, keep_dims, reduce_axes, output_shape),
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("do reduce infershape failed.")),
          return false);

    // when output is dynamic shape, will infer range
    if (output_shape.IsUnknownShape()) {
      if (!output_shape.IsUnknownDimNum()) {
        CHECK(!DoReduceInferRangeWithAxes(tensordesc_input_x, tensordesc_output, reduce_axes, keep_dims),
              VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("do reduce infer range failed.")),
              return false);
      }
      OP_LOGD(TbeGetName(op).c_str(), "infer output range end for dynamic output");
      return true;
    }
    OP_LOGD(TbeGetName(op).c_str(), "the output is not dynamic");
    PROFILING_PROTO_AFTER_INFER_SHAPE_REG();
    PROFILING_PROTO_END();
    return true;
  }

  CHECK(!DoReduceInferShapeWithoutAxes(op, tensordesc_input_x, tensordesc_output, axes_shape, keep_dims),
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("infer reduce range failed.")), return false);
  return true;
}

bool CommonReduceInferWithAttrAxes(const Operator &op, const int64_t input_x_idx, const int64_t output_idx,
                                   vector<int64_t> attr_axes, bool keep_dims) {
  PROFILING_PROTO_INIT(TbeGetName(op).c_str());
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_desc == nullptr, VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("invalid OpDesc.")),
        return false);
  auto tensordesc_input_x = op_desc->MutableInputDesc(input_x_idx);
  auto tensordesc_output = op_desc->MutableOutputDesc(output_idx);
  CHECK(tensordesc_input_x == nullptr || tensordesc_output == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("get mutable desc failed.")), return false);
  auto input_type = tensordesc_input_x->GetDataType();
  const GeShape &input_shape = tensordesc_input_x->MutableShape();
  tensordesc_output->SetDataType(input_type);

  PROFILING_PROTO_AFTER_GET_SHAPE_REG();
  // do infershape with const axes for static op
  GeShape &output_shape = tensordesc_output->MutableShape();
  CHECK(!DoReduceInfershapeWithAxes(input_shape, keep_dims, attr_axes, output_shape),
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("do reduce infershape failed.")), return false);

  // when output is dynamic shape, will infer range
  if (output_shape.IsUnknownShape()) {
    if (!output_shape.IsUnknownDimNum()) {
      CHECK(!DoReduceInferRangeWithAxes(tensordesc_input_x, tensordesc_output, attr_axes, keep_dims),
            VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("do reduce infer range failed.")),
            return false);
    }
    OP_LOGD(TbeGetName(op), "infer output range end for dynamic output");
    return true;
  }
  OP_LOGD(TbeGetName(op), "the output is not dynamic");
  PROFILING_PROTO_AFTER_INFER_SHAPE_REG();
  PROFILING_PROTO_END();
  return true;
}

}  // namespace reduce_ops
