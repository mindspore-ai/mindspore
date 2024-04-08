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
 * \file util.cpp
 * \brief
 */
#include "util.h"
#include <numeric>
#include <utility>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <algorithm>
#include <set>
#include "error_util.h"
#include "./vector_proto_profiling.h"
#include "op_common_util.h"

namespace ge {
using namespace std;

bool GetInputDataType(const ge::DataType &data_type, const std::vector<ge::DataType> &supportList) {
  std::vector<ge::DataType>::const_iterator supportIter = find(supportList.begin(), supportList.end(), data_type);
  if (supportIter == supportList.end()) {
    return false;
  }
  return true;
}

bool CheckInputDtypeAndShape(const Operator &op, const std::map<std::string, std::vector<DataType>> &inputTensorMap) {
  auto iter = inputTensorMap.begin();
  auto first_name = iter->first;
  auto first_shape_dims = op.GetInputDescByName(iter->first.c_str()).GetShape().GetDims();
  auto first_input_dtype = op.GetInputDescByName(iter->first.c_str()).GetDataType();
  for (; iter != inputTensorMap.end(); ++iter) {
    const TensorDesc input_desc = op.GetInputDescByName(iter->first.c_str());
    // check input dtype
    auto input_type = input_desc.GetDataType();
    if (input_type != first_input_dtype) {
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op),
        OtherErrMsg(ConcatString("the op type of param ", iter->first, " must equal with param ", first_name)));
      return false;
    }
    auto dims = input_desc.GetShape().GetDims();
    if (dims != first_shape_dims) {
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op),
        OtherErrMsg(ConcatString("the op shape of param ", iter->first, " must equal with param ", first_name)));
      return false;
    }
  }
  return true;
}

bool CheckInputDataType(const Operator &op, const std::string &input_name,
                        const std::vector<ge::DataType> &support_list) {
  bool valid = false;
  DataType input_type = op.GetInputDescByName(input_name.c_str()).GetDataType();
  do {
    const auto &found_list = find(support_list.begin(), support_list.end(), input_type);

    if (found_list == support_list.end()) {
      break;
    }

    const auto &found_map = DTYPE_STR_MAP.find(input_type);
    if (found_map == DTYPE_STR_MAP.end()) {
      break;
    }

    valid = true;
  } while (0);

  if (!valid) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
      TbeGetName(op), OtherErrMsg(ConcatString("The op do not support the dtype", GeDataTypeToString(input_type))));
    return false;
  }

  return true;
}

bool CheckTwoInputDtypeSame(const Operator &op, const string &input_name1, const string &input_name2) {
  DataType input_type_x1 = op.GetInputDesc(input_name1).GetDataType();
  DataType input_type_x2 = op.GetInputDesc(input_name2).GetDataType();
  if (input_type_x1 != input_type_x2) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
      TbeGetName(op), OtherErrMsg(ConcatString("The ", TbeGetName(op),
                                               " op dtype is not same, type1:", GeDataTypeToString(input_type_x1),
                                               ", type2:", GeDataTypeToString(input_type_x2))));
    return false;
  }

  return true;
}

bool CheckInputDtypeSame(const Operator &op, const std::vector<std::string> &input_names) {
  auto first_name = input_names.begin();
  auto first_input_dtype = op.GetInputDescByName((*first_name).c_str()).GetDataType();
  for (const string &input_name : input_names) {
    const TensorDesc input_desc = op.GetInputDescByName(input_name.c_str());
    auto input_dtype = input_desc.GetDataType();
    if (input_dtype != first_input_dtype) {
      auto error_ms = ConcatString("dtype of inputs must be same, ", input_name, ":", GeDataTypeToString(input_dtype),
                                   ", ", (*first_name), ":", GeDataTypeToString(first_input_dtype), ".");
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg(error_ms));
      return false;
    }
  }
  return true;
}

bool CheckInputsShapeDtypeSame(const Operator &op, const std::vector<std::string> &input_names) {
  auto first_input_name = input_names.begin();
  auto first_input_des = op.GetInputDescByName((*first_input_name).c_str());
  auto input_name = first_input_name;
  for (++input_name; input_name != input_names.end(); ++input_name) {
    auto input_des = op.GetInputDescByName((*first_input_name).c_str());
    if (input_des.GetDataType() != first_input_des.GetDataType() ||
        input_des.GetShape().GetDims() != first_input_des.GetShape().GetDims()) {
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), OtherErrMsg(ConcatString("the dtype and shape of param ", first_input_name->c_str(),
                                                 " must be same as param ", input_name->c_str())));
      return false;
    }
  }

  return true;
}

bool TwoShapeAndRangeBroadcastIntegration(const Operator &op, std::vector<int64_t> &dimVec,
                                          std::vector<std::pair<int64_t, int64_t>> &Vec_range,
                                          std::vector<int64_t> dims, std::vector<std::pair<int64_t, int64_t>> range,
                                          const string &input_name1, const string &input_name2) {
  if (dimVec.size() < dims.size()) {
    std::vector<int64_t> dimsTmp = dimVec;
    dimVec = dims;
    dims = dimsTmp;
    std::vector<std::pair<int64_t, int64_t>> range_temp = Vec_range;
    Vec_range = range;
    range = range_temp;
  }
  if (dimVec.size() != dims.size()) {
    int dec = static_cast<int>(dimVec.size() - dims.size());
    for (int i = 0; i < dec; i++) {
      dims.insert(dims.begin(), static_cast<int64_t>(1));
    }
  }
  for (size_t i = 0; i < dimVec.size(); i++) {
    CHECK((dimVec[i] != dims[i]) && (dimVec[i] != 1) && (dims[i] != 1) && (dimVec[i] != -1) && (dims[i] != -1),
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
            TbeGetName(op),
            OtherErrMsg(ConcatString("The ", TbeGetName(op), "'s dimensions does not match the broadcast rule(",
                                     dimVec[i], dims[i], ")."))),
          return false);
  }
  dimVec = TwoBroadcastShape(dimVec, dims);
  if (IsUnknown(dimVec)) {
    MakeUpShapeRange(dims, range);
    Vec_range = TwoShapeAndRangeBroadcast(dimVec, Vec_range, range);
  }
  return true;
}

std::vector<int64_t> TwoBroadcastShape(const std::vector<int64_t> &dimsX, const std::vector<int64_t> &dimsY) {
  std::vector<int64_t> dimVec;
  // when not dynamic case, do infer shape only
  if (!IsUnknown(dimsY) && !IsUnknown(dimsX)) {
    for (size_t i = 0; i < dimsX.size(); i++) {
      int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
      dims = (dimsY[i] == 0 || dimsX[i] == 0) ? 0 : dims;
      dimVec.push_back(dims);
    }
    return dimVec;
  }
  // dynamic case
  for (size_t i = 0; i < dimsX.size(); i++) {
    if ((dimsX[i] == -1) && (dimsY[i] != -1)) {
      if (dimsY[i] > 1) {
        int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
        dimVec.push_back(dims);
      } else if (dimsY[i] == 1) {
        int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
        dimVec.push_back(dims);
        dimVec[i] = -1;
      } else if ((dimsY[i] == 0) || (dimsX[i] == 0)) {
        dimVec.push_back(0);
      }
    } else if ((dimsX[i] != -1) && (dimsY[i] == -1)) {
      if (dimsX[i] > 1) {
        int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
        dimVec.push_back(dims);
      } else if (dimsX[i] == 0) {
        dimVec.push_back(0);
      } else if (dimsX[i] == 1) {
        int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
        dimVec.push_back(dims);
        dimVec[i] = -1;
      }
    } else {
      if ((dimsX[i] == -1) && (dimsY[i] == -1)) {
        int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
        dimVec.push_back(dims);
        dimVec[i] = -1;
      } else {
        if (dimsY[i] == 0 || dimsX[i] == 0) {
          dimVec.push_back(0);
        } else {
          int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
          dimVec.push_back(dims);
        }
      }
    }
  }
  return dimVec;
}

std::vector<std::pair<int64_t, int64_t>> TwoShapeAndRangeBroadcast(
  const std::vector<int64_t> &dims_out, const std::vector<std::pair<int64_t, int64_t>> &shape_range_x,
  std::vector<std::pair<int64_t, int64_t>> &shape_range_y) {
  size_t size_shape_out = dims_out.size();
  std::vector<std::pair<int64_t, int64_t>> out_range;
  if (!IsUnknownRankShape(dims_out)) {
    while (shape_range_x.size() > shape_range_y.size()) {
      shape_range_y.insert(shape_range_y.begin(), std::pair<int64_t, int64_t>(1, 1));
    }
    for (size_t i = 0; i < size_shape_out; i++) {
      if (dims_out[i] != -1) {
        out_range.push_back(std::pair<int64_t, int64_t>(dims_out[i], dims_out[i]));
        continue;
      }
      if (i < shape_range_x.size() && i < shape_range_y.size()) {
        if (shape_range_x[i].second == -1 && shape_range_y[i].second == 1) {
          out_range.push_back(std::pair<int64_t, int64_t>(1, -1));
        } else if (shape_range_x[i].second == 1 && shape_range_y[i].second == -1) {
          out_range.push_back(std::pair<int64_t, int64_t>(1, -1));
        } else if (shape_range_x[i].first == 1 || shape_range_y[i].first == 1) {
          // one shape size maybe 1, so will support broadcast
          // first_range == max first
          int64_t first_range = std::max(shape_range_x[i].first, shape_range_y[i].first);
          int64_t second_range = shape_range_x[i].first == 1 ? shape_range_y[i].second : shape_range_x[i].second;
          if (shape_range_x[i].first == 1 && shape_range_y[i].first == 1) {
            second_range = std::max(shape_range_x[i].second, shape_range_y[i].second);
            second_range = (shape_range_x[i].second == -1 || shape_range_y[i].second == -1) ? -1 : second_range;
          }
          out_range.push_back(std::pair<int64_t, int64_t>(first_range, second_range));
        } else {
          // no 1 in range.first, mean no broadcast for range
          // get intersect range
          int64_t first_range = std::max(shape_range_x[i].first, shape_range_y[i].first);
          int64_t second_range = std::min(shape_range_x[i].second, shape_range_y[i].second);
          second_range = (shape_range_x[i].second == -1 || shape_range_y[i].second == -1)
                           ? std::max(shape_range_x[i].second, shape_range_y[i].second)
                           : second_range;
          out_range.push_back(std::pair<int64_t, int64_t>(first_range, second_range));
        }
      }
    }
  }
  return out_range;
}

bool InferBroadcastshapeForStatic(const Shape &shape_x, const Shape &shape_y, Shape &shape_output) {
  auto shape_x_len = shape_x.GetDimNum();
  auto shape_y_len = shape_y.GetDimNum();

  OP_LOGI("BroadcastInfer", "input1 shape is: %s, input2 shape is: %s.", to_string(shape_x).c_str(),
          to_string(shape_y).c_str());
  std::vector<int64_t> output_shape;
  if (shape_x_len >= shape_y_len) {
    // when inputx len >= inputy len
    // input_x = [128, 128, 128] Vs input_y = [128]
    auto len_sub = shape_x_len - shape_y_len;
    for (size_t i = 0; i < len_sub; i++) {
      (void)output_shape.emplace_back(shape_x.GetDim(i));
    }
    for (size_t i = 0; i < shape_y_len; i++) {
      int64_t dim_size = std::max(shape_x.GetDim(len_sub + i), shape_y.GetDim(i));
      // if one dim is 0, the output dim is 0
      dim_size = (shape_x.GetDim(len_sub + i) == 0 || shape_y.GetDim(i) == 0) ? 0 : dim_size;
      (void)output_shape.emplace_back(dim_size);
    }
  } else {
    // when inputx len < inputy len
    // input_x = [128] Vs input_y = [128, 128, 128]
    auto len_sub = shape_y_len - shape_x_len;
    for (size_t i = 0; i < len_sub; i++) {
      (void)output_shape.emplace_back(shape_y.GetDim(i));
    }
    for (size_t i = 0; i < shape_x_len; i++) {
      int64_t dim_size = std::max(shape_y.GetDim(len_sub + i), shape_x.GetDim(i));
      // if one dim is 0, the output dim is 0
      dim_size = (shape_y.GetDim(len_sub + i) == 0 || shape_x.GetDim(i) == 0) ? 0 : dim_size;
      (void)output_shape.emplace_back(dim_size);
    }
  }
  shape_output = Shape(output_shape);
  OP_LOGI("BroadcastInfer", "output1 shape is: %s.", to_string(shape_output).c_str());
  return true;
}

bool InferShapeAndTypeTwoInOneOutBroadcast(Operator &op, const string &input_name1, const string &input_name2,
                                           const string &output_name, bool &is_dynamic) {
  PROFILING_PROTO_INIT(TbeGetName(op).c_str());
  DataType input_dtype = op.GetInputDesc(input_name1).GetDataType();

  // output Desc
  auto tensordesc_output = op.GetOutputDesc(output_name);
  tensordesc_output.SetDataType(input_dtype);

  ge::Shape shapeX = op.GetInputDesc(input_name1).GetShape();
  ge::Shape shapeY = op.GetInputDesc(input_name2).GetShape();
  OP_LOGI(TbeGetName(op).c_str(), "shape %s: %s, shape %s: %s.", input_name1.c_str(), to_string(shapeX).c_str(),
          input_name2.c_str(), to_string(shapeY).c_str());
  std::vector<int64_t> dimsX = shapeX.GetDims();
  std::vector<int64_t> dimsY = shapeY.GetDims();
  PROFILING_PROTO_AFTER_GET_SHAPE_REG();
  // swap based on shape size
  if (dimsX.size() < dimsY.size()) {
    std::vector<int64_t> dimsTmp = dimsX;
    dimsX = dimsY;
    dimsY = dimsTmp;
  }

  // unknown rank
  if (IsUnknownRankShape(dimsX) || IsUnknownRankShape(dimsY)) {
    tensordesc_output.SetShape(ge::Shape(UNKNOWN_RANK));
    OP_LOGI(TbeGetName(op).c_str(), "output shape is: %s, output dtype is:%d.",
            to_string(ge::Shape(UNKNOWN_RANK)).c_str(), input_dtype);
    is_dynamic = false;
    op.UpdateOutputDesc(output_name, tensordesc_output);
    return true;
  }

  // pad 1 for small shape
  if (dimsX.size() != dimsY.size()) {
    int dec = static_cast<int>(dimsX.size() - dimsY.size());
    for (int i = 0; i < dec; i++) {
      dimsY.insert(dimsY.begin(), (int64_t)1);
    }
  }

  // when not dynamic case, do infer shape only
  if (!IsUnKnownShape(dimsY) && !IsUnKnownShape(dimsX)) {
    std::vector<int64_t> dimVec(dimsX.size(), 0);
    for (size_t i = 0; i < dimsX.size(); i++) {
      dimVec[i] = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
      dimVec[i] = (dimsY[i] == 0 || dimsX[i] == 0) ? 0 : dimVec[i];
    }

    PROFILING_PROTO_AFTER_INFER_SHAPE_REG();
    tensordesc_output.SetShape(ge::Shape(dimVec));
    is_dynamic = false;
    op.UpdateOutputDesc(output_name, tensordesc_output);
    PROFILING_PROTO_END();
    return true;
  }

  std::vector<int64_t> dimVec;
  // dynamic case
  for (size_t i = 0; i < dimsX.size(); i++) {
    CHECK((dimsX[i] != dimsY[i]) && (dimsX[i] != 1) && (dimsY[i] != 1) && (dimsX[i] != -1) && (dimsY[i] != -1),
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
            TbeGetName(op),
            OtherErrMsg(ConcatString("The ", TbeGetName(op), "'s dimensions does not match the broadcast rule(",
                                     dimsX[i], dimsY[i], ")."))),
          return false);

    if ((dimsX[i] == -1) && (dimsY[i] != -1)) {
      if (dimsY[i] > 1) {
        int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
        dimVec.push_back(dims);
      } else if (dimsY[i] == 1) {
        int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
        dimVec.push_back(dims);
        dimVec[i] = -1;
      } else if ((dimsY[i] == 0) || (dimsX[i] == 0)) {
        dimVec.push_back(-1);
      }
    } else if ((dimsX[i] != -1) && (dimsY[i] == -1)) {
      if (dimsX[i] > 1) {
        int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
        dimVec.push_back(dims);
      } else if (dimsX[i] == 0) {
        dimVec.push_back(-1);
      } else if (dimsX[i] == 1) {
        int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
        dimVec.push_back(dims);
        dimVec[i] = -1;
      }
    } else {
      if ((dimsX[i] == -1) && (dimsY[i] == -1)) {
        int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
        dimVec.push_back(dims);
        dimVec[i] = -1;
      } else {
        if (dimsY[i] == 0 || dimsX[i] == 0) {
          dimVec.push_back(0);
        } else {
          int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
          dimVec.push_back(dims);
        }
      }
    }
  }
  ge::Shape outputShape = ge::Shape(dimVec);
  tensordesc_output.SetShape(outputShape);

  OP_LOGI(TbeGetName(op).c_str(), "output shape is: %s, output dtype is:%s.", to_string(outputShape).c_str(),
          GeDataTypeToString(input_dtype).c_str());
  is_dynamic = IsUnknown(dimVec);
  if (is_dynamic) {
    if (!InferShapeRangeTwoInOneOutBroadcast(op, input_name1, input_name2, output_name)) {
      return false;
    }
  }
  op.UpdateOutputDesc(output_name, tensordesc_output);
  return true;
}

bool InferShapeAndTypeTwoInOneOutBroadcast(Operator &op, const string &input_name1, const string &input_name2,
                                           const string &output_name) {
  DataType input_dtype = op.GetInputDesc(input_name1).GetDataType();

  auto tensordesc_output = op.GetOutputDesc(output_name);

  ge::Shape shapeX = op.GetInputDesc(input_name1).GetShape();
  ge::Shape shapeY = op.GetInputDesc(input_name2).GetShape();
  OP_LOGI(TbeGetName(op).c_str(), "shape %s: %s, shape %s: %s.", input_name1.c_str(), to_string(shapeX).c_str(),
          input_name2.c_str(), to_string(shapeY).c_str());
  std::vector<int64_t> dimsX = shapeX.GetDims();
  std::vector<int64_t> dimsY = shapeY.GetDims();
  // swap based on shape size
  if (dimsX.size() < dimsY.size()) {
    std::vector<int64_t> dimsTmp = dimsX;
    dimsX = dimsY;
    dimsY = dimsTmp;
  }

  std::vector<int64_t> dimVec;

  // unknown rank
  if (IsUnknownRankShape(dimsX) || IsUnknownRankShape(dimsY)) {
    tensordesc_output.SetShape(ge::Shape(UNKNOWN_RANK));
    tensordesc_output.SetDataType(input_dtype);
    OP_LOGI(TbeGetName(op).c_str(), "output shape is: %s, output dtype is:%d.",
            to_string(ge::Shape(UNKNOWN_RANK)).c_str(), input_dtype);
    op.UpdateOutputDesc(output_name, tensordesc_output);
    return true;
  }

  // pad 1 for small shape
  if (dimsX.size() != dimsY.size()) {
    int dec = static_cast<int>(dimsX.size() - dimsY.size());
    for (int i = 0; i < dec; i++) {
      dimsY.insert(dimsY.begin(), (int64_t)1);
    }
  }

  for (size_t i = 0; i < dimsX.size(); i++) {
    CHECK((dimsX[i] != dimsY[i]) && (dimsX[i] != 1) && (dimsY[i] != 1) && (dimsX[i] != -1) && (dimsY[i] != -1),
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
            TbeGetName(op),
            OtherErrMsg(ConcatString("The ", TbeGetName(op), "'s dimensions does not match the broadcast rule(",
                                     dimsX[i], dimsY[i], ")."))),
          return false);

    if ((dimsX[i] == -1) && (dimsY[i] != -1)) {
      if (dimsY[i] > 1) {
        int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
        dimVec.push_back(dims);
      } else if (dimsY[i] == 1) {
        int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
        dimVec.push_back(dims);
        dimVec[i] = -1;
      } else if ((dimsY[i] == 0) || (dimsX[i] == 0)) {
        dimVec.push_back(0);
      }
    } else if ((dimsX[i] != -1) && (dimsY[i] == -1)) {
      if (dimsX[i] > 1) {
        int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
        dimVec.push_back(dims);
      } else if (dimsX[i] == 0) {
        dimVec.push_back(0);
      } else if (dimsX[i] == 1) {
        int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
        dimVec.push_back(dims);
        dimVec[i] = -1;
      }
    } else {
      if ((dimsX[i] == -1) && (dimsY[i] == -1)) {
        int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
        dimVec.push_back(dims);
        dimVec[i] = -1;
      } else {
        if (dimsY[i] == 0 || dimsX[i] == 0) {
          dimVec.push_back(0);
        } else {
          int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
          dimVec.push_back(dims);
        }
      }
    }
  }
  ge::Shape outputShape = ge::Shape(dimVec);

  tensordesc_output.SetShape(outputShape);
  tensordesc_output.SetDataType(input_dtype);
  OP_LOGI(TbeGetName(op).c_str(), "output shape is: %s, output dtype is:%s.", to_string(outputShape).c_str(),
          GeDataTypeToString(input_dtype).c_str());
  op.UpdateOutputDesc(output_name, tensordesc_output);

  return true;
}

std::string ToFormatString(ge::Format format) { return GeFormatToString(format); }

static void AddToOutputRange(std::vector<std::pair<int64_t, int64_t>> &out_range,
                             const std::pair<int64_t, int64_t> &shape_range_x,
                             const std::pair<int64_t, int64_t> &shape_range_y) {
  // first_range == max first
  int64_t first_range =
    (shape_range_x.first * shape_range_y.first == 0) ? 0 : std::max(shape_range_x.first, shape_range_y.first);

  if (shape_range_x.second * shape_range_y.second == -1) {
    out_range.push_back(std::pair<int64_t, int64_t>(first_range, -1));
  } else if (shape_range_x.first == 1 && shape_range_y.first == 1) {
    int64_t second_range = (shape_range_x.second == -1 || shape_range_y.second == -1)
                             ? -1
                             : std::max(shape_range_x.second, shape_range_y.second);
    out_range.push_back(std::pair<int64_t, int64_t>(first_range, second_range));
  } else if (shape_range_x.first == 1 || shape_range_y.first == 1) {
    // one shape size maybe 1, so will support broadcast
    int64_t second_range = shape_range_x.first == 1 ? shape_range_y.second : shape_range_x.second;
    out_range.push_back(std::pair<int64_t, int64_t>(first_range, second_range));
  } else {
    // no 1 in range.first, mean no broadcast for range
    // get intersect range
    int64_t second_range = std::min(shape_range_x.second, shape_range_y.second);
    second_range = (shape_range_x.second == -1 || shape_range_y.second == -1)
                     ? std::max(shape_range_x.second, shape_range_y.second)
                     : second_range;
    out_range.push_back(std::pair<int64_t, int64_t>(first_range, second_range));
  }
}

bool InferShapeRangeTwoInOneOutBroadcast(Operator &op, const string &input_name1, const string &input_name2,
                                         const string &output_name) {
  ge::Shape shape_x = op.GetInputDesc(input_name1).GetShape();
  ge::Shape shape_y = op.GetInputDesc(input_name2).GetShape();

  std::vector<int64_t> dims_x = shape_x.GetDims();
  std::vector<int64_t> dims_y = shape_y.GetDims();

  std::vector<std::pair<int64_t, int64_t>> shape_range_x;
  op.GetInputDesc(input_name1).GetShapeRange(shape_range_x);
  std::vector<std::pair<int64_t, int64_t>> shape_range_y;
  op.GetInputDesc(input_name2).GetShapeRange(shape_range_y);

  MakeUpShapeRange(dims_x, shape_range_x);
  MakeUpShapeRange(dims_y, shape_range_y);

  ge::Shape shape_out = op.GetOutputDesc(output_name).GetShape();
  std::vector<int64_t> dims_out = shape_out.GetDims();
  size_t size_shape_out = dims_out.size();

  std::vector<std::pair<int64_t, int64_t>> out_range;

  if (!IsUnknownRankShape(dims_out)) {
    // shape switch by shape dim size
    if (dims_x.size() < dims_y.size()) {
      std::vector<int64_t> dims_tmp = dims_x;
      dims_x = dims_y;
      dims_y = dims_tmp;

      std::vector<std::pair<int64_t, int64_t>> range_temp = shape_range_x;
      shape_range_x = shape_range_y;
      shape_range_y = range_temp;
    }

    while (dims_x.size() > shape_range_y.size()) {
      shape_range_y.insert(shape_range_y.begin(), std::pair<int64_t, int64_t>(1, 1));
    }

    for (size_t i = 0; i < size_shape_out; i++) {
      if (dims_out[i] != -1) {
        out_range.push_back(std::pair<int64_t, int64_t>(dims_out[i], dims_out[i]));
        continue;
      }
      if (i < shape_range_x.size() && i < shape_range_y.size()) {
        AddToOutputRange(out_range, shape_range_x[i], shape_range_y[i]);
      }
    }
  }
  OP_LOGI(TbeGetName(op).c_str(), "elewise out range is %s", to_string(out_range).c_str());
  auto tensor_out = op.GetOutputDesc(output_name);
  tensor_out.SetShapeRange(out_range);
  op.UpdateOutputDesc(output_name, tensor_out);

  return true;
}

bool GetInputDataType(const ge::DataType &dataType, const std::vector<ge::DataType> &supportList, std::string &dType) {
  std::vector<ge::DataType>::const_iterator supportIter = find(supportList.begin(), supportList.end(), dataType);
  if (supportIter == supportList.end()) {
    return false;
  }

  std::map<ge::DataType, std::string>::const_iterator totalIter = DTYPE_STR_MAP.find(dataType);
  if (totalIter == DTYPE_STR_MAP.end()) {
    return false;
  }

  dType = totalIter->second;
  return true;
}

bool CheckInputDataType(const Operator &op, std::string *data_type, const std::string &input_name,
                        const std::vector<ge::DataType> &supportList) {
  DataType input_type = op.GetInputDescByName(input_name.c_str()).GetDataType();
  if (false == GetInputDataType(input_type, supportList, *data_type)) {
    LOG_ERROR("[ERROR]op [%s] [%s] do not supported dtype [%s]!\n", TbeGetName(op).c_str(), input_name.c_str(),
              data_type->c_str());
    return false;
  }
  return true;
}

bool GetConstValue(const ge::Operator &op, const std::string &key_name, float &attr_value) {
  if (ge::GRAPH_SUCCESS != op.GetAttr(key_name.c_str(), attr_value)) {
    LOG_ERROR("[ERROR]op [%s] GetOpAttr [%s] failed!\n", TbeGetName(op).c_str(), key_name.c_str());
    return false;
  }
  return true;
}

bool GetConstValue(const ge::Operator &op, const std::string &key_name, int64_t &attr_value) {
  if (ge::GRAPH_SUCCESS != op.GetAttr(key_name.c_str(), attr_value)) {
    LOG_ERROR("[ERROR]op [%s] GetOpAttr [%s] failed!\n", TbeGetName(op).c_str(), key_name.c_str());
    return false;
  }
  return true;
}

bool GetConstValue(const ge::Operator &op, const std::string &key_name, bool &attr_value) {
  if (ge::GRAPH_SUCCESS != op.GetAttr(key_name.c_str(), attr_value)) {
    LOG_ERROR("[ERROR]op [%s] GetOpAttr [%s] failed!\n", TbeGetName(op).c_str(), key_name.c_str());
    return false;
  }
  return true;
}

bool GetConstValue(const ge::Operator &op, const std::string &key_name, std::vector<int32_t> &attr_value) {
  if (ge::GRAPH_SUCCESS != op.GetAttr(key_name.c_str(), attr_value)) {
    LOG_ERROR("[ERROR]op [%s] GetOpAttr [%s] failed!\n", TbeGetName(op).c_str(), key_name.c_str());
    return false;
  }
  return true;
}

template <typename T>
static std::vector<int64_t> GetConstIntData(const uint8_t *const_data, size_t data_size) {
  size_t size = data_size / sizeof(T);
  std::vector<int64_t> result(size);
  const T *data = reinterpret_cast<const T *>(const_data);
  for (size_t i = 0; i < size; i++) {
    result[i] = *(data + i);
  }

  return result;
}

bool GetConstIntData(const Tensor &data, DataType data_type, std::vector<int64_t> &const_values) {
  using std::placeholders::_1;
  using std::placeholders::_2;
  const std::map<DataType, std::function<std::vector<int64_t>(const uint8_t *, size_t)>> type_call_map = {
    {DT_INT8, std::bind(GetConstIntData<int8_t>, _1, _2)},
    {DT_INT16, std::bind(GetConstIntData<int16_t>, _1, _2)},
    {DT_INT32, std::bind(GetConstIntData<int32_t>, _1, _2)},
    {DT_INT64, std::bind(GetConstIntData<int64_t>, _1, _2)},
  };

  auto found = type_call_map.find(data_type);
  if (found == type_call_map.end()) {
    USER_GE_LOGE("[ERROR]GetConstIntData is not support data_type[%s]!", GeDataTypeToString(data_type).c_str());
    return false;
  }

  const_values = found->second(data.GetData(), data.GetSize());

  return true;
}

bool GetConstValue(const Operator &op, const Tensor &const_tensor, const DataType &dtype,
                   std::vector<int64_t> &const_data) {
  CHECK(dtype != ge::DT_INT32 && dtype != ge::DT_INT64,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("not support this type")), return false);
  if (dtype == ge::DT_INT32) {
    const int32_t *const_data_ptr = reinterpret_cast<const int32_t *>(const_tensor.GetData());
    size_t size = const_tensor.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back(static_cast<int32_t>(*(const_data_ptr + i)));
      OP_LOGD(TbeGetName(op).c_str(), "const data int32 fusion pass ====== %d",
              static_cast<int32_t>(*(const_data_ptr + i)));
    }
  } else if (dtype == ge::DT_INT64) {
    const int64_t *const_data_ptr = reinterpret_cast<const int64_t *>(const_tensor.GetData());
    size_t size = const_tensor.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back(static_cast<int64_t>(*(const_data_ptr + i)));
      OP_LOGD(TbeGetName(op).c_str(), "const data int64 fusion pass ====== %ld",
              static_cast<int64_t>(*(const_data_ptr + i)));
    }
  }
  return true;
}

bool GetConstValue(const Operator &op, const Tensor &const_tensor, const DataType &dtype,
                   std::vector<uint64_t> &const_data) {
  size_t size = 0;
  CHECK(dtype != ge::DT_UINT64,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("not support this type")), return false);
  const uint64_t *const_data_ptr = reinterpret_cast<const uint64_t *>(const_tensor.GetData());
  size = const_tensor.GetSize() / sizeof(uint64_t);
  for (size_t i = 0; i < size; ++i) {
    const_data.push_back(static_cast<uint64_t>(*(const_data_ptr + i)));
    OP_LOGD(TbeGetName(op).c_str(), "const data uint64 fusion pass, const_data[%lu]",
            static_cast<uint64_t>(*(const_data_ptr + i)));
  }
  return true;
}

bool GetScalerValue(const Operator &op, const Tensor &const_tensor, const DataType &dtype, std::int64_t &const_data) {
  if (dtype == ge::DT_INT32) {
    const int32_t *const_data_ptr = reinterpret_cast<const int32_t *>(const_tensor.GetData());
    const_data = static_cast<int32_t>(*const_data_ptr);
  } else if (dtype == ge::DT_INT64) {
    const int64_t *const_data_ptr = reinterpret_cast<const int64_t *>(const_tensor.GetData());
    const_data = static_cast<int64_t>(*const_data_ptr);
  } else {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg(ConcatString("not support this type:", dtype)));
    return false;
  }
  return true;
}

std::string to_string(const std::vector<int64_t> &shape) { return ops::to_string(shape); }

std::string to_string(const ge::Shape &shape) { return to_string(shape.GetDims()); }

std::string to_string(const std::vector<std::pair<int64_t, int64_t>> &ranges) { return ops::to_string(ranges); }

static std::map<ge::DataType, std::string> kDataTypeToStringMap = {{ge::DataType::DT_FLOAT, "float"},
                                                                   {ge::DataType::DT_FLOAT16, "float16"},
                                                                   {ge::DataType::DT_INT8, "int8"},
                                                                   {ge::DataType::DT_INT16, "int16"},
                                                                   {ge::DataType::DT_UINT16, "uint16"},
                                                                   {ge::DataType::DT_UINT8, "uint8"},
                                                                   {ge::DataType::DT_INT32, "int32"},
                                                                   {ge::DataType::DT_INT64, "int64"},
                                                                   {ge::DataType::DT_UINT32, "uint32"},
                                                                   {ge::DataType::DT_UINT64, "uint64"},
                                                                   {ge::DataType::DT_BOOL, "bool"},
                                                                   {ge::DataType::DT_DOUBLE, "double"},
                                                                   {ge::DataType::DT_STRING, "string"},
                                                                   {ge::DataType::DT_DUAL_SUB_INT8, "dual_sub_int8"},
                                                                   {ge::DataType::DT_DUAL_SUB_UINT8, "dual_sub_uint8"},
                                                                   {ge::DataType::DT_COMPLEX64, "complex64"},
                                                                   {ge::DataType::DT_COMPLEX128, "complex128"},
                                                                   {ge::DataType::DT_DUAL, "dual"},
                                                                   {ge::DataType::DT_QINT8, "qint8"},
                                                                   {ge::DataType::DT_QINT16, "qint16"},
                                                                   {ge::DataType::DT_QINT32, "qint32"},
                                                                   {ge::DataType::DT_QUINT8, "quint8"},
                                                                   {ge::DataType::DT_QUINT16, "quint16"},
                                                                   {ge::DataType::DT_RESOURCE, "resource"},
                                                                   {ge::DataType::DT_STRING_REF, "string ref"},
                                                                   {ge::DataType::DT_VARIANT, "dt_variant"},
                                                                   {ge::DataType::DT_UNDEFINED, "undefined"},
                                                                   {ge::DataType::DT_INT4, "int4"},
                                                                   {ge::DataType::DT_UINT1, "uint1"},
                                                                   {ge::DataType::DT_INT2, "int2"},
                                                                   {ge::DataType::DT_UINT2, "uint2"},
                                                                   {ge::DataType::DT_COMPLEX32, "complex32"},
                                                                   {ge::DataType::DT_BF16, "bf16"}};

static std::map<ge::Format, std::string> kFormatToStringMap = {
  {ge::Format::FORMAT_NCHW, "NCHW"},
  {ge::Format::FORMAT_NHWC, "NHWC"},
  {ge::Format::FORMAT_ND, "Nd"},
  {ge::Format::FORMAT_NC1HWC0, "NC1HWC0"},
  {ge::Format::FORMAT_FRACTAL_Z, "FRACTAL_Z"},
  {ge::Format::FORMAT_NC1C0HWPAD, "NC1C0HWPAD"},
  {ge::Format::FORMAT_NHWC1C0, "NHWC1C0"},
  {ge::Format::FORMAT_FSR_NCHW, "FSR_NCHW"},
  {ge::Format::FORMAT_FRACTAL_DECONV, "FRACTAL_DECONV"},
  {ge::Format::FORMAT_C1HWNC0, "C1HWNC0"},
  {ge::Format::FORMAT_FRACTAL_DECONV_TRANSPOSE, "FRACTAL_DECONV_TRANSPOSE"},
  {ge::Format::FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS, "FRACTAL_DECONV_SP_STRIDE_TRANS"},
  {ge::Format::FORMAT_NC1HWC0_C04, "NC1HWC0_C04"},
  {ge::Format::FORMAT_FRACTAL_Z_C04, "FRACTAL_Z_C04"},
  {ge::Format::FORMAT_CHWN, "CHWN"},
  {ge::Format::FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS, "FRACTAL_DECONV_SP_STRIDE8_TRANS"},
  {ge::Format::FORMAT_HWCN, "HWCN"},
  {ge::Format::FORMAT_NC1KHKWHWC0, "NC1KHKWHWC0"},
  {ge::Format::FORMAT_BN_WEIGHT, "BN_WEIGHT"},
  {ge::Format::FORMAT_FILTER_HWCK, "FILTER_HWCK"},
  {ge::Format::FORMAT_HASHTABLE_LOOKUP_LOOKUPS, "HASHTABLE_LOOKUP_LOOKUPS"},
  {ge::Format::FORMAT_HASHTABLE_LOOKUP_KEYS, "HASHTABLE_LOOKUP_KEYS"},
  {ge::Format::FORMAT_HASHTABLE_LOOKUP_VALUE, "HASHTABLE_LOOKUP_VALUE"},
  {ge::Format::FORMAT_HASHTABLE_LOOKUP_OUTPUT, "HASHTABLE_LOOKUP_OUTPUT"},
  {ge::Format::FORMAT_HASHTABLE_LOOKUP_HITS, "HASHTABLE_LOOKUP_HITS"},
  {ge::Format::FORMAT_C1HWNCoC0, "C1HWNCoC0"},
  {ge::Format::FORMAT_MD, "MD"},
  {ge::Format::FORMAT_NDHWC, "NDHWC"},
  {ge::Format::FORMAT_FRACTAL_ZZ, "FRACTAL_ZZ"},
  {ge::Format::FORMAT_FRACTAL_NZ, "FRACTAL_NZ"},
  {ge::Format::FORMAT_NCDHW, "NCDHW"},
  {ge::Format::FORMAT_DHWCN, "DHWCN"},
  {ge::Format::FORMAT_NDC1HWC0, "NDC1HWC0"},
  {ge::Format::FORMAT_FRACTAL_Z_3D, "FRACTAL_Z_3D"},
  {ge::Format::FORMAT_CN, "CN"},
  {ge::Format::FORMAT_NC, "NC"},
  {ge::Format::FORMAT_DHWNC, "DHWNC"},
  {ge::Format::FORMAT_FRACTAL_Z_3D_TRANSPOSE, "FRACTAL_Z_3D_TRANSPOSE"},
  {ge::Format::FORMAT_FRACTAL_ZN_LSTM, "FRACTAL_ZN_LSTM"},
  {ge::Format::FORMAT_FRACTAL_Z_G, "FRACTAL_Z_G"},
  {ge::Format::FORMAT_RESERVED, "RESERVED"},
  {ge::Format::FORMAT_ALL, "ALL"},
  {ge::Format::FORMAT_NULL, "NULL"},
  {ge::Format::FORMAT_ND_RNN_BIAS, "ND_RNN_BIAS"},
  {ge::Format::FORMAT_FRACTAL_ZN_RNN, "FRACTAL_ZN_RNN"},
  {ge::Format::FORMAT_NYUV, "NYUV"},
  {ge::Format::FORMAT_NYUV_A, "NYUV_A"},
  {ge::Format::FORMAT_NCL, "NCL"}};

std::string GeDataTypeToString(const ge::DataType datatype) {
  auto iter = kDataTypeToStringMap.find(datatype);
  if (iter != kDataTypeToStringMap.end()) {
    return iter->second;
  }
  return "";
}

std::string GeFormatToString(const ge::Format format) {
  auto iter = kFormatToStringMap.find(format);
  if (iter != kFormatToStringMap.end()) {
    return iter->second;
  }
  return "";
}

bool IsEmptyTensor(const std::vector<int64_t> &dims) {
  if (dims.size() == 1 && dims[0] == 0) {
    return true;
  } else {
    return false;
  }
}

bool IsUnknownRank(const Operator &op, const std::string &tensor_name, const std::string &types) {
  TensorDesc tensor_desc;
  if (types == "input") {
    tensor_desc = op.GetInputDesc(tensor_name);
  } else if (types == "output") {
    tensor_desc = op.GetOutputDesc(tensor_name);
  } else {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                        OtherErrMsg(ConcatString("invalid params:", types, " of types to judge.")));
    return false;
  }

  std::vector<int64_t> shape_vec = tensor_desc.GetShape().GetDims();
  if (shape_vec.size() == 1 && shape_vec[0] == INPUT_NEGATIVE_NUM2) {
    return true;
  }
  return false;
}

bool IsUnknownRankShape(const std::vector<int64_t> &shape_vec) {
  if (shape_vec.size() == 1 && shape_vec[0] == ge::UNKNOWN_DIM_NUM) {
    return true;
  }
  return false;
}

bool IsUnknownRankShape(const Shape &input_shape) {
  auto dims = input_shape.GetDims();
  return (dims.size() == 1UL) && (dims[0UL] == UNKNOWN_DIM_NUM);
}

bool IsUnKnownShape(const std::vector<int64_t> &shape_vec) {
  auto found = find(shape_vec.begin(), shape_vec.end(), -1);
  return found != shape_vec.end();
}

bool IsUnknown(const std::vector<int64_t> &shape_vec) {
  return (IsUnKnownShape(shape_vec) || IsUnknownRankShape(shape_vec));
}

bool IsUnknownVec(std::vector<int64_t> &shape_vec) {
  std::vector<int64_t>::iterator it_shape = find(shape_vec.begin(), shape_vec.end(), -1);
  if (it_shape == shape_vec.end()) {
    return false;
  } else {
    return true;
  }
}

void MakeUpShapeRange(const std::vector<int64_t> &shape, std::vector<std::pair<int64_t, int64_t>> &range) {
  if (IsUnknownRankShape(shape)) {
    return;
  }

  if (range.empty()) {
    for (size_t i = 0; i < shape.size(); i++) {
      if (shape[i] == -1) {
        range.push_back(std::pair<int64_t, int64_t>(0, -1));
      } else {
        range.push_back(std::pair<int64_t, int64_t>(shape[i], shape[i]));
      }
    }
  }
}

void MakeUpShapeRange(const ge::Shape &shape, std::vector<std::pair<int64_t, int64_t>> &range) {
  if (IsUnknownRankShape(shape)) {
    return;
  }

  if (range.empty()) {
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
      int64_t dim = shape.GetDim(i);
      if (dim == -1) {
        range.push_back(std::pair<int64_t, int64_t>(0, -1));
      } else {
        range.push_back(std::pair<int64_t, int64_t>(dim, dim));
      }
    }
  }
}

std::string DataTypeToStringDesc(const ge::DataType &dataType) {
  std::map<ge::DataType, std::string>::const_iterator totalIter = DTYPE_STR_MAP.find(dataType);
  if (totalIter == DTYPE_STR_MAP.end()) {
    return "UNDEFINED";
  }
  return totalIter->second;
}

bool OneInOneOutDynamicInfer(Operator &op, const std::string &input_name,
                             const std::vector<std::string> &output_name_list) {
  // get input desc
  PROFILING_PROTO_INIT(TbeGetName(op).c_str());
  auto input_desc = op.GetInputDesc(input_name);
  vector<int64_t> input_shape = input_desc.GetShape().GetDims();
  DataType input_dtype = input_desc.GetDataType();

  if (IsUnknown(input_shape)) {
    std::vector<std::pair<int64_t, int64_t>> input_range;
    input_desc.GetShapeRange(input_range);
    MakeUpShapeRange(input_shape, input_range);

    auto output_desc = op.GetOutputDesc(0);
    for (const string &output_name : output_name_list) {
      output_desc = op.GetOutputDesc(output_name);
      output_desc.SetShape(Shape(input_shape));
      output_desc.SetOriginShape(Shape(input_shape));
      output_desc.SetShapeRange(input_range);
      output_desc.SetDataType(input_dtype);
      op.UpdateOutputDesc(output_name, output_desc);
    }
  } else {
    auto output_desc = op.GetOutputDesc(0);
    PROFILING_PROTO_AFTER_GET_SHAPE_REG();
    PROFILING_PROTO_AFTER_INFER_SHAPE_REG();
    for (const string &output_name : output_name_list) {
      output_desc = op.GetOutputDesc(output_name);
      output_desc.SetShape(Shape(input_shape));
      output_desc.SetDataType(input_dtype);
      op.UpdateOutputDesc(output_name, output_desc);
    }
    PROFILING_PROTO_END();
  }
  return true;
}

void FixShapeRangeWithDims(const std::vector<int64_t> &dims, std::vector<int64_t> &shape_1,
                           std::vector<int64_t> &shape_2, std::vector<std::pair<int64_t, int64_t>> &range_1,
                           std::vector<std::pair<int64_t, int64_t>> &range_2) {
  MakeUpShapeRange(shape_1, range_1);
  MakeUpShapeRange(shape_2, range_2);
  bool is_all_fix = dims.empty();

  if (shape_1 == UNKNOWN_RANK && shape_2 == UNKNOWN_RANK) {
    return;
  }
  if (shape_1 == UNKNOWN_RANK) {
    shape_1 = shape_2;
    range_1 = range_2;
    return;
  }
  if (shape_2 == UNKNOWN_RANK) {
    shape_2 = shape_1;
    range_2 = range_1;
    return;
  }
  if ((shape_1.size() != shape_2.size()) || (range_1.size() != range_2.size())) {
    return;
  }
  auto loop_size = is_all_fix ? shape_1.size() : dims.size();
  for (size_t i = 0; i < loop_size; i++) {
    auto dim_num = is_all_fix ? i : dims[i];
    if (shape_1[dim_num] != -1) {
      shape_2[dim_num] = shape_1[dim_num];
      range_1[dim_num] = std::pair<int64_t, int64_t>(shape_1[dim_num], shape_1[dim_num]);
      range_2[dim_num] = std::pair<int64_t, int64_t>(shape_1[dim_num], shape_1[dim_num]);
      continue;
    }
    if (shape_2[dim_num] != -1) {
      shape_1[dim_num] = shape_2[dim_num];
      range_1[dim_num] = std::pair<int64_t, int64_t>(shape_2[dim_num], shape_2[dim_num]);
      range_2[dim_num] = std::pair<int64_t, int64_t>(shape_2[dim_num], shape_2[dim_num]);
      continue;
    }
    // both the dim in shape1 and shape2 are -1
    auto range_1_min = range_1[dim_num].first;
    auto range_2_min = range_2[dim_num].first;
    auto range_1_max = range_1[dim_num].second;
    auto range_2_max = range_2[dim_num].second;
    auto range_fisrt = range_1_min > range_2_min ? range_1_min : range_2_min;
    auto range_second_min = range_1_max > range_2_max ? range_2_max : range_1_max;
    auto range_second_max = range_1_max > range_2_max ? range_1_max : range_2_max;
    range_second_min = range_second_min == -1 ? range_second_max : range_second_min;
    range_1[dim_num] = std::pair<int64_t, int64_t>(range_fisrt, range_second_min);
    range_2[dim_num] = std::pair<int64_t, int64_t>(range_fisrt, range_second_min);
  }
}

bool TwoInOneOutDynamicInferNoBroadcast(Operator &op, const string &input1_name, const string &input2_name,
                                        const std::vector<string> &output_name_list) {
  // get input1 desc
  auto input1_desc = op.GetInputDesc(input1_name);
  vector<int64_t> input1_shape = input1_desc.GetShape().GetDims();
  DataType input_dtype = input1_desc.GetDataType();

  // get input2 desc
  auto input2_desc = op.GetInputDesc(input2_name);
  vector<int64_t> input2_shape = input2_desc.GetShape().GetDims();

  if (IsUnknown(input1_shape) || IsUnknown(input2_shape)) {
    std::vector<std::pair<int64_t, int64_t>> input1_range;
    input1_desc.GetShapeRange(input1_range);
    std::vector<std::pair<int64_t, int64_t>> input2_range;
    input2_desc.GetShapeRange(input2_range);

    vector<int64_t> dim_size = {};
    FixShapeRangeWithDims(dim_size, input1_shape, input2_shape, input1_range, input2_range);

    // update output desc
    for (const string &output_name : output_name_list) {
      auto output_desc = op.GetOutputDesc(output_name);
      output_desc.SetShape(Shape(input1_shape));
      output_desc.SetOriginShape(Shape(input1_shape));
      output_desc.SetShapeRange(input1_range);
      output_desc.SetDataType(input_dtype);
      op.UpdateOutputDesc(output_name, output_desc);
    }
  } else {
    for (const string &output_name : output_name_list) {
      auto output_desc = op.GetOutputDesc(output_name);
      output_desc.SetShape(Shape(input1_shape));
      output_desc.SetDataType(input_dtype);
      op.UpdateOutputDesc(output_name, output_desc);
    }
  }
  return true;
}

bool IsEmptyTensor(TensorDesc tensor_desc) { return IsEmptyTensor(tensor_desc.GetShape()); }

bool IsEmptyTensor(const Shape &ge_shape) {
  bool is_empty = false;
  for (const auto &dim : ge_shape.GetDims()) {
    if (dim == 0) {
      is_empty = true;
      break;
    }
  }
  return is_empty;
}

bool IsUnknownShape(const ge::Shape &shape) {
  const auto &dims = shape.GetDims();
  return std::any_of(dims.begin(), dims.end(),
                     [](const int64_t &dim) { return (dim == UNKNOWN_DIM) || (dim == UNKNOWN_DIM_NUM); });
}

bool IsUnknownDimNum(const ge::Shape &shape) {
  const auto &dims = shape.GetDims();
  return (dims.size() == 1UL) && (dims[0UL] == UNKNOWN_DIM_NUM);
}

bool IsScalar(const ge::Shape &shape) {
  const auto &dims = shape.GetDims();
  return dims.empty();
}

void SetOpInferDepends(Operator &op, const std::vector<std::string> &depend_names) {
  op.SetAttr(ATTR_NAME_OP_INFER_DEPENDS, depend_names);
}

void SetIsUnknownDimNum(ge::Shape &shape) {
  std::vector<int64_t> dims(1UL, UNKNOWN_DIM_NUM);
  dims[0UL] = UNKNOWN_DIM_NUM;
  shape = ge::Shape(dims);
}

namespace array_ops {
// If not overflow return true
bool CheckInt64MulOverflow(int64_t a, int64_t b) {
  if (a > 0) {
    if (b > 0) {
      if (a > (INT64_MAX / b)) {
        return false;
      }
    } else {
      if (b < (INT64_MIN / a)) {
        return false;
      }
    }
  } else {
    if (b > 0) {
      if (a < (INT64_MIN / b)) {
        return false;
      }
    } else {
      if ((a != 0) && (b < (INT64_MAX / a))) {
        return false;
      }
    }
  }

  return true;
}

int64_t CalcMaxElementsCount(const Operator &op, const std::vector<std::pair<int64_t, int64_t>> &x_shape_range,
                             const Shape &x_shape) {
  int64_t max_elements_count = 1;
  auto x_shape_size = x_shape.GetShapeSize();
  if (x_shape_size > 0) {
    // when known dim, x_shape_size is max_elements_count
    max_elements_count = x_shape_size;
  } else {
    // unknown dim
    if (x_shape_range.empty()) {
      max_elements_count = -1;
    }
    for (const auto &x_range_i : x_shape_range) {
      if (x_range_i.second <= 0) {
        max_elements_count = -1;
        break;
      }
      if (array_ops::CheckInt64MulOverflow(max_elements_count, x_range_i.second)) {
        max_elements_count *= x_range_i.second;
      } else {
        max_elements_count = -1;
        break;
      }
    }
  }

  return max_elements_count;
}

void GenerateWorstYShapeAndYShapeRange(int64_t y_rank, int64_t max_elements_count,
                                       std::vector<std::pair<int64_t, int64_t>> &y_shape_range, Shape &y_shape) {
  y_shape = Shape(std::vector<int64_t>(y_rank, UNKNOWN_DIM));
  y_shape_range.clear();
  for (int64_t i = 0; i < y_rank; ++i) {
    y_shape_range.emplace_back(std::pair<int64_t, int64_t>(1, max_elements_count));
  }
}

bool RepairAndCheckRange(const std::vector<std::pair<int64_t, int64_t>> &x_shape_range,
                         std::vector<std::pair<int64_t, int64_t>> &value_range) {
  bool has_zero_in_range = false;
  for (auto &range_i : value_range) {
    if (range_i.first < 0) {
      range_i.first = 1;
    }
    if (range_i.second < 0) {
      range_i.second = -1;
    }
    if (range_i.first == 0) {
      has_zero_in_range = true;
    }
  }

  for (auto &range_i : x_shape_range) {
    if (range_i.first == 0) {
      has_zero_in_range = true;
      break;
    }
  }
  return has_zero_in_range;
}

void InferShapeRangeForEmptyTensor(int64_t y_rank, int64_t max_elements_count,
                                   const std::vector<std::pair<int64_t, int64_t>> &value_range,
                                   std::vector<std::pair<int64_t, int64_t>> &y_shape_range, Shape &y_shape) {
  y_shape_range = value_range;
  int64_t known_dims_product = 1;
  std::vector<int64_t> y_dims = y_shape.GetDims();
  for (int64_t i = 0; i < y_rank; ++i) {
    if (y_shape_range[i].first == y_shape_range[i].second) {
      y_dims[i] = y_shape_range[i].first;
      if (max_elements_count != -1 && y_dims[i] != 0) {
        known_dims_product *= y_dims[i];
      }
    }
  }
  y_shape = Shape(y_dims);

  if (known_dims_product != 1) {
    auto cur_dim_max_elements_count = (max_elements_count - 1) / known_dims_product + 1;
    for (int64_t i = 0; i < y_rank; ++i) {
      if (y_dims[i] == -1) {
        if (y_shape_range[i].second != -1) {
          y_shape_range[i].second = std::min(cur_dim_max_elements_count, y_shape_range[i].second);
        } else {
          y_shape_range[i].second = cur_dim_max_elements_count;
        }
      }
    }
  }
}

void UpdateDimsAndShapeRange(const Operator &op, int64_t max_elements_count,
                             const std::vector<std::pair<int64_t, int64_t>> &value_range, std::vector<int64_t> &y_dims,
                             std::vector<std::pair<int64_t, int64_t>> &y_shape_range) {
  size_t y_rank = y_dims.size();
  for (size_t i = 0; i < y_rank; ++i) {
    if (value_range[i].first == value_range[i].second) {
      y_dims[i] = value_range[i].first;
      y_shape_range[i] = std::pair<int64_t, int64_t>(y_dims[i], y_dims[i]);
    } else {
      if (max_elements_count == -1) {
        // while max_elements_count = -1, y shape range i is always value_range[i].second;
        y_shape_range[i] = std::pair<int64_t, int64_t>(value_range[i].first, value_range[i].second);
        continue;
      }
      int64_t other_dims_range_lower_boundary = 1;
      for (size_t j = 0; j < y_rank; ++j) {
        if (i != j) {
          other_dims_range_lower_boundary *= value_range[j].first;
        }
      }
      int64_t cur_dim_range_max = (max_elements_count - 1) / other_dims_range_lower_boundary + 1;
      if (value_range[i].second > 0) {
        cur_dim_range_max = std::min(cur_dim_range_max, value_range[i].second);
      }
      y_shape_range[i] = std::pair<int64_t, int64_t>(value_range[i].first, cur_dim_range_max);
    }
  }
}

int64_t CalculateMaxInputDims(const std::vector<std::pair<int64_t, int64_t>> &x_range, const Operator &op) {
  int64_t max_input_dims = 1;
  for (const auto &pair : x_range) {
    if (pair.second < 0) {
      max_input_dims = -1;
      break;
    }

    if (array_ops::CheckInt64MulOverflow(max_input_dims, pair.second)) {
      max_input_dims *= pair.second;
    } else {
      max_input_dims = INT64_MAX;
      GE_OP_LOGW(TbeGetName(op).c_str(), "Range Infer out of int64 max!Do set int64max!");
      break;
    }
  }
  return max_input_dims;
}
}  // namespace array_ops

bool IsSliceUnknownShape(const std::vector<int64_t> &dim_vec, const int64_t &begin, const int64_t &end) {
  if (begin < 0 || end >= static_cast<int64_t>(dim_vec.size())) {
    GE_OP_LOGE("FlattenV2", "index is out of range");
    return false;
  }
  for (int64_t i = begin; i < end + 1; i++) {
    if (dim_vec[i] == -1) {
      return true;
    }
  }
  return false;
}

void SetOpInferDepends(Operator &op, const std::vector<std::string> &depend_names);
}  // namespace ge
