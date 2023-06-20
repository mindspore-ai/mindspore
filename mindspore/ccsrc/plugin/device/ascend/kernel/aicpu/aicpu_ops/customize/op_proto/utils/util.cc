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
#include "graph/utils/type_utils.h"
#include "axis_util.h"

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
      TbeGetName(op),
      OtherErrMsg(ConcatString("The op do not support the dtype", ge::TypeUtils::DataTypeToSerialString(input_type))));
    return false;
  }

  return true;
}

bool CheckTwoInputDtypeSame(const Operator &op, const string &input_name1, const string &input_name2) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_desc == nullptr || op_desc->MutableInputDesc(input_name1) == nullptr ||
          op_desc->MutableInputDesc(input_name2) == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("invalid OpDesc.")), return false);

  DataType input_type_x1 = op_desc->MutableInputDesc(input_name1)->GetDataType();
  DataType input_type_x2 = op_desc->MutableInputDesc(input_name2)->GetDataType();
  if (input_type_x1 != input_type_x2) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
      TbeGetName(op), OtherErrMsg(ConcatString("The ", TbeGetName(op), " op dtype is not same, type1:",
                                               ge::TypeUtils::DataTypeToSerialString(input_type_x1),
                                               ", type2:", ge::TypeUtils::DataTypeToSerialString(input_type_x2))));
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
      auto error_ms = ConcatString("dtype of inputs must be same, ", input_name, ":",
                                   ge::TypeUtils::DataTypeToSerialString(input_dtype), ", ", (*first_name), ":",
                                   ge::TypeUtils::DataTypeToSerialString(first_input_dtype), ".");
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
    int dec = dimVec.size() - dims.size();
    for (int i = 0; i < dec; i++) {
      dims.insert(dims.begin(), (int64_t)1);
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

bool InferBroadcastshapeForStatic(const GeShape &shape_x, const GeShape &shape_y, GeShape &shape_output) {
  auto shape_x_len = shape_x.GetDimNum();
  auto shape_y_len = shape_y.GetDimNum();

  OP_LOGI("BroadcastInfer", "input1 shape is: %s, input2 shape is: %s.", to_string(shape_x).c_str(),
          to_string(shape_y).c_str());
  if (shape_x_len >= shape_y_len) {
    // when inputx len >= inputy len
    // input_x = [128, 128, 128] Vs input_y = [128]
    auto len_sub = shape_x_len - shape_y_len;
    shape_output.SetDimNum(shape_x_len);
    for (size_t i = 0; i < len_sub; i++) {
      shape_output.SetDim(i, shape_x.GetDim(i));
    }
    for (size_t i = 0; i < shape_y_len; i++) {
      int64_t dim_size = std::max(shape_x.GetDim(len_sub + i), shape_y.GetDim(i));
      // if one dim is 0, the output dim is 0
      dim_size = (shape_x.GetDim(len_sub + i) == 0 || shape_y.GetDim(i) == 0) ? 0 : dim_size;
      shape_output.SetDim(len_sub + i, dim_size);
    }
  } else {
    // when inputx len < inputy len
    // input_x = [128] Vs input_y = [128, 128, 128]
    auto len_sub = shape_y_len - shape_x_len;
    shape_output.SetDimNum(shape_y_len);
    for (size_t i = 0; i < len_sub; i++) {
      shape_output.SetDim(i, shape_y.GetDim(i));
    }
    for (size_t i = 0; i < shape_x_len; i++) {
      int64_t dim_size = std::max(shape_y.GetDim(len_sub + i), shape_x.GetDim(i));
      // if one dim is 0, the output dim is 0
      dim_size = (shape_y.GetDim(len_sub + i) == 0 || shape_x.GetDim(i) == 0) ? 0 : dim_size;
      shape_output.SetDim(len_sub + i, dim_size);
    }
  }
  OP_LOGI("BroadcastInfer", "output1 shape is: %s.", to_string(shape_output).c_str());
  return true;
}

bool InferShapeAndTypeTwoInOneOutBroadcast(Operator &op, const int64_t &input_idx_1, const int64_t &input_idx_2,
                                           const int64_t &output_idx, bool &is_dynamic) {
  PROFILING_PROTO_INIT(TbeGetName(op).c_str());
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_desc == nullptr, VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("invalid OpDesc.")),
        return false);
  auto tensordesc_input_1 = op_desc->MutableInputDesc(input_idx_1);
  auto tensordesc_input_2 = op_desc->MutableInputDesc(input_idx_2);
  auto tensordesc_output = op_desc->MutableOutputDesc(output_idx);
  CHECK(tensordesc_output == nullptr || tensordesc_input_1 == nullptr || tensordesc_input_2 == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("invalid tensordesc.")), return false);

  // set output
  DataType input_dtype = tensordesc_input_1->GetDataType();
  tensordesc_output->SetDataType(input_dtype);

  const GeShape &shape_x = tensordesc_input_1->MutableShape();
  const GeShape &shape_y = tensordesc_input_2->MutableShape();
  OP_LOGI(TbeGetName(op).c_str(), "shape_1: %s, shape_2: %s.", to_string(shape_x).c_str(), to_string(shape_y).c_str());

  PROFILING_PROTO_AFTER_GET_SHAPE_REG();

  if (IsUnknownRankShape(shape_x) || IsUnknownRankShape(shape_y)) {
    OP_LOGI(TbeGetName(op).c_str(), "do unknownrank infershape for Broadcast");
    tensordesc_output->SetShape(GeShape(UNKNOWN_RANK));
    is_dynamic = false;
    return true;
  }
  // do static infershape start
  if ((!shape_x.IsUnknownShape()) && (!shape_y.IsUnknownShape())) {
    OP_LOGI(TbeGetName(op).c_str(), "do static infershape for Broadcast");
    GeShape &shape_output = tensordesc_output->MutableShape();
    CHECK(!InferBroadcastshapeForStatic(shape_x, shape_y, shape_output),
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("infershape failed.")), return false);
    OP_LOGI(TbeGetName(op).c_str(), "output shape is: %s, output dtype is:%s.", to_string(shape_output).c_str(),
            ge::TypeUtils::DataTypeToSerialString(input_dtype).c_str());
    is_dynamic = false;
    PROFILING_PROTO_AFTER_INFER_SHAPE_REG();
    PROFILING_PROTO_END();
    return true;
  }
  // do static infershape end

  // dynamic case
  auto input_name1 = op_desc->GetInputNameByIndex(input_idx_1);
  auto input_name2 = op_desc->GetInputNameByIndex(input_idx_2);
  auto output_name = op_desc->GetOutputNameByIndex(output_idx);
  OP_LOGI(TbeGetName(op).c_str(), "get the input name by idx is  %s vs %s vs %s", input_name1.c_str(),
          input_name2.c_str(), output_name.c_str());
  return InferShapeAndTypeTwoInOneOutBroadcast(op, input_name1, input_name2, output_name, is_dynamic);
}

bool InferShapeAndTypeTwoInOneOutBroadcast(Operator &op, const string &input_name1, const string &input_name2,
                                           const string &output_name, bool &is_dynamic) {
  PROFILING_PROTO_INIT(TbeGetName(op).c_str());
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_desc == nullptr || op_desc->MutableOutputDesc(output_name) == nullptr ||
          op_desc->MutableInputDesc(input_name1) == nullptr || op_desc->MutableInputDesc(input_name2) == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("invalid OpDesc.")), return false);

  DataType input_dtype = op_desc->MutableInputDesc(input_name1)->GetDataType();

  // output Desc
  GeTensorDescPtr tensordesc_output = op_desc->MutableOutputDesc(output_name);
  tensordesc_output->SetDataType(input_dtype);

  ge::GeShape shapeX = op_desc->MutableInputDesc(input_name1)->GetShape();
  ge::GeShape shapeY = op_desc->MutableInputDesc(input_name2)->GetShape();
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
    tensordesc_output->SetShape(ge::GeShape(UNKNOWN_RANK));
    OP_LOGI(TbeGetName(op).c_str(), "output shape is: %s, output dtype is:%d.",
            to_string(ge::Shape(UNKNOWN_RANK)).c_str(), input_dtype);
    is_dynamic = false;
    return true;
  }

  // pad 1 for small shape
  if (dimsX.size() != dimsY.size()) {
    int dec = dimsX.size() - dimsY.size();
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
    tensordesc_output->SetShape(ge::GeShape(dimVec));
    is_dynamic = false;
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
  ge::GeShape outputShape = ge::GeShape(dimVec);
  tensordesc_output->SetShape(outputShape);

  OP_LOGI(TbeGetName(op).c_str(), "output shape is: %s, output dtype is:%s.", to_string(outputShape).c_str(),
          ge::TypeUtils::DataTypeToSerialString(input_dtype).c_str());
  is_dynamic = IsUnknown(dimVec);
  if (is_dynamic) {
    if (!InferShapeRangeTwoInOneOutBroadcast(op, input_name1, input_name2, output_name)) {
      return false;
    }
  }
  return true;
}

bool InferShapeAndTypeTwoInOneOutBroadcast(Operator &op, const string &input_name1, const string &input_name2,
                                           const string &output_name) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_desc == nullptr || op_desc->MutableInputDesc(input_name1) == nullptr ||
          op_desc->MutableOutputDesc(output_name) == nullptr || op_desc->MutableInputDesc(input_name2) == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("invalid OpDesc.")), return false);

  DataType input_dtype = op_desc->MutableInputDesc(input_name1)->GetDataType();

  GeTensorDescPtr tensordesc_output = op_desc->MutableOutputDesc(output_name);

  ge::GeShape shapeX = op_desc->MutableInputDesc(input_name1)->GetShape();
  ge::GeShape shapeY = op_desc->MutableInputDesc(input_name2)->GetShape();
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
    tensordesc_output->SetShape(ge::GeShape(UNKNOWN_RANK));
    tensordesc_output->SetDataType(input_dtype);
    OP_LOGI(TbeGetName(op).c_str(), "output shape is: %s, output dtype is:%d.",
            to_string(ge::Shape(UNKNOWN_RANK)).c_str(), input_dtype);
    return true;
  }

  // pad 1 for small shape
  if (dimsX.size() != dimsY.size()) {
    int dec = dimsX.size() - dimsY.size();
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
  ge::GeShape outputShape = ge::GeShape(dimVec);

  tensordesc_output->SetShape(outputShape);
  tensordesc_output->SetDataType(input_dtype);
  OP_LOGI(TbeGetName(op).c_str(), "output shape is: %s, output dtype is:%s.", to_string(outputShape).c_str(),
          ge::TypeUtils::DataTypeToSerialString(input_dtype).c_str());

  return true;
}
static bool BroadCastShapeToOutShape(const GeShape &shape, GeShape &shape_output) {
  size_t shape_len = shape.GetDimNum();
  size_t shape_y_len = shape_output.GetDimNum();
  int64_t dim1, dim2;

  if (shape_len > shape_y_len) {
    shape_output.SetDimNum(shape_len);
    size_t len_sub = shape_len - shape_y_len;
    for (size_t i = shape_y_len; i > 0; i--) {
      dim1 = shape.GetDim(len_sub + i - 1);
      dim2 = shape_output.GetDim(i - 1);
      // if dim1 equal dim2 do nothing, else dim1<=1 or dim2<=1
      if (dim1 != dim2) {
        /* column is dim1, row is dim2, matrix value is broadcast(dim1, dim2)
        dim   -1    0     1    d2
        -1    -1    0     -1   -1
        0     0     0     0    0
        1     -1    0     1    d2
        d1    -1    0     d1   E
        */
        if ((dim1 > 1) && (dim2 > 1)) {
          std::string err_msg = shape_output.ToString() + " and " + shape.ToString() + " cannot broadcast";
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT("BroadCastShapeToOutShape", err_msg);
          return false;
        }
        if (dim2 == 0) {
          dim1 = 0;
        }
        if (dim1 == 1) {
          dim1 = dim2;
        }
        if ((dim1 > 1) && (dim2 == -1)) {
          dim1 = -1;
        }
      }
      shape_output.SetDim(len_sub + i - 1, dim1);
    }
    for (size_t i = 0; i < len_sub; i++) {
      shape_output.SetDim(i, shape.GetDim(i));
    }
  } else {
    auto len_sub = shape_y_len - shape_len;
    for (size_t i = 0; i < shape_len; i++) {
      dim1 = shape_output.GetDim(len_sub + i);
      dim2 = shape.GetDim(i);
      // if dim1 equal dim2 do nothing, else dim1<=1 or dim2<=1
      if (dim1 != dim2) {
        /* column is dim1, row is dim2, matrix value is broadcast(dim1, dim2)
        dim   -1    0     1    d2
        -1    -1    0     -1   -1
        0     0     0     0    0
        1     -1    0     1    d2
        d1    -1    0     d1   E
        */
        if ((dim1 > 1) && (dim2 > 1)) {
          std::string err_msg = shape_output.ToString() + " and " + shape.ToString() + " cannot broadcast";
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT("BroadCastShapeToOutShape", err_msg);
          return false;
        }
        if (dim2 == 0) {
          dim1 = 0;
        }
        if (dim1 == 1) {
          dim1 = dim2;
        }
        if ((dim1 > 1) && (dim2 == -1)) {
          dim1 = -1;
        }
        shape_output.SetDim(len_sub + i, dim1);
      }
    }
  }
  return true;
}

bool InferShapeAndTypeBroadcast(Operator &op, std::vector<int64_t> input_idxs, const int64_t &output_idx,
                                bool &is_dynamic) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_desc == nullptr, VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("invalid OpDesc.")),
        return false);
  GeTensorDescPtr tensordesc_output = op_desc->MutableOutputDesc(output_idx);
  CHECK(tensordesc_output == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("invalid tensordesc.")), return false);
  GeTensorDescPtr input_desc = op_desc->MutableInputDesc(input_idxs[0]);
  CHECK(input_desc == nullptr, VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("invalid tensordesc.")),
        return false);
  // do static infershape start
  OP_LOGI(TbeGetName(op).c_str(), "do static infershape for Broadcast");
  GeShape &shape_output = tensordesc_output->MutableShape();
  shape_output = input_desc->MutableShape();
  // set output
  DataType input_dtype = input_desc->GetDataType();
  tensordesc_output->SetDataType(input_dtype);
  if (shape_output.IsUnknownDimNum()) {
    OP_LOGI(TbeGetName(op), "do unknownrank infershape for Broadcast");
    tensordesc_output->SetShape(shape_output);
    return true;
  }
  is_dynamic = shape_output.IsUnknownShape();
  size_t input_count = input_idxs.size();
  for (size_t i = 1; i < input_count; i++) {
    input_desc = op_desc->MutableInputDesc(input_idxs[i]);
    CHECK(input_desc == nullptr,
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("invalid tensordesc.")), return false);
    const ge::GeShape &shape = input_desc->MutableShape();
    if (shape.IsUnknownDimNum()) {
      OP_LOGI(TbeGetName(op), "do unknownrank infershape for Broadcast");
      tensordesc_output->SetShape(shape);
      is_dynamic = false;
      return true;
    }
    if ((!is_dynamic) && shape.IsUnknownShape()) {
      OP_LOGI(TbeGetName(op), "unknown shape_%zu: %s.", i, to_string(shape).c_str());
      is_dynamic = true;
    }
    OP_LOGI(TbeGetName(op), "shape_%zu: %s.", i, to_string(shape).c_str());
    if (!BroadCastShapeToOutShape(shape, shape_output)) {
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("BroadCastShapeToOutShape error."));
      return false;
    }
    OP_LOGI(TbeGetName(op), "shape_output%zu: %s.", i, to_string(shape_output).c_str());
  }
  return true;
}

static std::vector<int64_t> GetNewAxis4NDC1HWC0(std::size_t ori_shape_len, int64_t axis, const std::string &ori_format,
                                                bool reduce_mode) {
  string ori_format_upper = ori_format;
  transform(ori_format_upper.begin(), ori_format_upper.end(), ori_format_upper.begin(), ::toupper);

  if (ori_format_upper == "NDC1HWC0") {
    return {axis};
  }

  const int64_t n_axis = 0;
  const int64_t d_axis = 1;
  const int64_t c1_axis = 2;
  const int64_t h_axis = 3;
  const int64_t w_axis = 4;
  const int64_t c0_axis = 5;

  vector<int64_t> new_c_axis = {c1_axis};
  if (reduce_mode) {
    new_c_axis.push_back(c0_axis);
  }

  map<char, vector<int64_t>> new_format_axis_map = {
    {'N', {n_axis}}, {'C', new_c_axis}, {'H', {h_axis}}, {'W', {w_axis}}, {'D', {d_axis}},
  };

  int64_t non_negative_axis = axis;
  if (non_negative_axis < 0) {
    non_negative_axis += ori_shape_len;
  }

  if (static_cast<size_t>(non_negative_axis) < ori_format_upper.length()) {
    const char axis_dim_name = ori_format_upper[non_negative_axis];
    auto found = new_format_axis_map.find(axis_dim_name);
    if (found != new_format_axis_map.end()) {
      return found->second;
    }
  }

  return {};
}

static std::vector<int64_t> GetNewAxis4NC1HWC0(std::size_t ori_shape_len, int64_t axis, const std::string &ori_format,
                                               bool reduce_mode) {
  string ori_format_upper = ori_format;
  transform(ori_format_upper.begin(), ori_format_upper.end(), ori_format_upper.begin(), ::toupper);

  if (ori_format_upper == "NC1HWC0") {
    return {axis};
  }

  const int64_t n_axis = 0;
  const int64_t c1_axis = 1;
  const int64_t h_axis = 2;
  const int64_t w_axis = 3;
  const int64_t c0_axis = 4;

  vector<int64_t> new_c_axis = {c1_axis};
  if (reduce_mode) {
    new_c_axis.push_back(c0_axis);
  }

  map<char, vector<int64_t>> new_format_axis_map = {
    {'N', {n_axis}},
    {'C', new_c_axis},
    {'H', {h_axis}},
    {'W', {w_axis}},
  };

  int64_t non_negative_axis = axis;
  if (non_negative_axis < 0) {
    non_negative_axis += ori_shape_len;
  }

  if (static_cast<size_t>(non_negative_axis) < ori_format_upper.length()) {
    const char axis_dim_name = ori_format_upper[non_negative_axis];
    auto found = new_format_axis_map.find(axis_dim_name);
    if (found != new_format_axis_map.end()) {
      return found->second;
    }
  }

  return {};
}

// FRACTAL_NZ means: [A, B, ..., C, D] -> [A, B, ..., ceil(D//16), ceil(C//16), 16, 16]
static std::vector<int64_t> GetNewAxis4FRACTAL_NZ(std::size_t ori_shape_len, int64_t axis,
                                                  const std::string &ori_format, bool reduce_mode) {
  string ori_format_upper = ori_format;
  transform(ori_format_upper.begin(), ori_format_upper.end(), ori_format_upper.begin(), ::toupper);

  if (ori_format_upper == "FRACTAL_NZ") {
    return {axis};
  }

  int64_t non_negative_axis = axis;
  if (non_negative_axis < 0) {
    non_negative_axis += ori_shape_len;
  }

  if (static_cast<size_t>(non_negative_axis) >= ori_shape_len) {
    return {};
  }

  int64_t new_shape_len = max<int64_t>(static_cast<int64_t>(ori_shape_len), 2) + 2;
  if (static_cast<size_t>(non_negative_axis) == ori_shape_len - 1) {
    if (!reduce_mode) {
      return {new_shape_len - 4};
    }

    return {new_shape_len - 4, new_shape_len - 1};
  }

  if (static_cast<size_t>(non_negative_axis) == ori_shape_len - NUM_VALUE2) {
    if (!reduce_mode) {
      return {new_shape_len - 3};
    }

    return {new_shape_len - 3, new_shape_len - 2};
  }

  return {non_negative_axis};
}

std::vector<int64_t> GetNewAxis4NewFormat(std::size_t ori_shape_len, int64_t axis, const std::string &ori_format,
                                          const std::string &new_format, bool reduce_mode) {
  string ori_format_upper = ori_format;
  string new_format_upper = new_format;
  transform(ori_format_upper.begin(), ori_format_upper.end(), ori_format_upper.begin(), ::toupper);
  transform(new_format_upper.begin(), new_format_upper.end(), new_format_upper.begin(), ::toupper);
  if (ori_format_upper == new_format_upper) {
    return {axis};
  }
  using transform_func = std::function<std::vector<int64_t>(std::size_t, int64_t, const std::string &, bool)>;

  // FRACTAL_NZ means: [A, B, ..., C, D] -> [A, B, ..., ceil(D//16), ceil(C//16), 16, 16]
  const map<string, transform_func> format_transform_func = {
    {"NDC1HWC0", GetNewAxis4NDC1HWC0}, {"NC1HWC0", GetNewAxis4NC1HWC0}, {"FRACTAL_NZ", GetNewAxis4FRACTAL_NZ}};

  auto found = format_transform_func.find(new_format_upper);
  if (found != format_transform_func.end()) {
    return found->second(ori_shape_len, axis, ori_format, reduce_mode);
  }

  return {};
}

std::string ToFormatString(ge::Format format) { return ge::TypeUtils::FormatToSerialString(format); }

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
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_desc == nullptr || op_desc->MutableInputDesc(input_name1) == nullptr ||
          op_desc->MutableOutputDesc(output_name) == nullptr || op_desc->MutableInputDesc(input_name2) == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("invalid OpDesc.")), return false);

  ge::GeShape shape_x = op_desc->MutableInputDesc(input_name1)->GetShape();
  ge::GeShape shape_y = op_desc->MutableInputDesc(input_name2)->GetShape();

  std::vector<int64_t> dims_x = shape_x.GetDims();
  std::vector<int64_t> dims_y = shape_y.GetDims();

  std::vector<std::pair<int64_t, int64_t>> shape_range_x;
  op_desc->MutableInputDesc(input_name1)->GetShapeRange(shape_range_x);
  std::vector<std::pair<int64_t, int64_t>> shape_range_y;
  op_desc->MutableInputDesc(input_name2)->GetShapeRange(shape_range_y);

  MakeUpShapeRange(dims_x, shape_range_x);
  MakeUpShapeRange(dims_y, shape_range_y);

  ge::GeShape shape_out = op_desc->MutableOutputDesc(output_name)->GetShape();
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
  GeTensorDescPtr tensor_out = op_desc->MutableOutputDesc(output_name);
  tensor_out->SetShapeRange(out_range);

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
  T *data = (T *)const_data;
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
    USER_GE_LOGE("[ERROR]GetConstIntData is not support data_type[%s]!",
                 ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
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
    int32_t *const_data_ptr = (int32_t *)const_tensor.GetData();
    size_t size = const_tensor.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back((int32_t)((*(const_data_ptr + i))));
      OP_LOGD(TbeGetName(op).c_str(), "const data int32 fusion pass ====== %d", (int32_t)(*(const_data_ptr + i)));
    }
  } else if (dtype == ge::DT_INT64) {
    int64_t *const_data_ptr = (int64_t *)const_tensor.GetData();
    size_t size = const_tensor.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back(((int64_t)(*(const_data_ptr + i))));
      OP_LOGD(TbeGetName(op).c_str(), "const data int64 fusion pass ====== %ld", (int64_t)(*(const_data_ptr + i)));
    }
  }
  return true;
}

bool GetConstValue(const Operator &op, const Tensor &const_tensor, const DataType &dtype,
                   std::vector<uint64_t> &const_data) {
  size_t size = 0;
  CHECK(dtype != ge::DT_UINT64,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("not support this type")), return false);
  uint64_t *const_data_ptr = (uint64_t *)const_tensor.GetData();
  size = const_tensor.GetSize() / sizeof(uint64_t);
  for (size_t i = 0; i < size; ++i) {
    const_data.push_back((uint64_t)(*(const_data_ptr + i)));
    OP_LOGD(TbeGetName(op).c_str(), "const data uint64 fusion pass, const_data[%lu]",
            (uint64_t)(*(const_data_ptr + i)));
  }
  return true;
}

bool GetConstValue(const Operator &op, const GeTensor *const_tensor, const DataType &dtype,
                   std::vector<int64_t> &const_data) {
  size_t size = const_tensor->GetData().GetSize();
  void *data_ptr = (void *)const_tensor->GetData().GetData();
  CHECK(data_ptr == nullptr, VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("data is null.")),
        return false);

  CHECK(dtype != ge::DT_INT32 && dtype != ge::DT_INT64,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("const not support this type")), return false);
  if (dtype == ge::DT_INT32) {
    int32_t *const_data_ptr = reinterpret_cast<int32_t *>(data_ptr);
    size = size / sizeof(int32_t);
    for (size_t i = 0; i < size; i++) {
      const_data.push_back((int64_t)((int32_t)((*(const_data_ptr + i)))));
    }
  } else if (dtype == ge::DT_INT64) {
    int64_t *const_data_ptr = reinterpret_cast<int64_t *>(data_ptr);
    size = size / sizeof(int64_t);
    for (size_t i = 0; i < size; i++) {
      const_data.push_back((int64_t)((int64_t)((*(const_data_ptr + i)))));
    }
  }
  return true;
}

bool GetConstValue(const Operator &op, const GeTensorPtr &const_tensor, const DataType &dtype,
                   std::vector<int64_t> &const_data) {
  return GetConstValue(op, const_tensor.get(), dtype, const_data);
}

bool GetScalerValue(const Operator &op, const Tensor &const_tensor, const DataType &dtype, std::int64_t &const_data) {
  if (dtype == ge::DT_INT32) {
    int32_t *const_data_ptr = (int32_t *)const_tensor.GetData();
    const_data = (int32_t)(*const_data_ptr);
  } else if (dtype == ge::DT_INT64) {
    int64_t *const_data_ptr = (int64_t *)const_tensor.GetData();
    const_data = (int64_t)(*const_data_ptr);
  } else {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg(ConcatString("not support this type:", dtype)));
    return false;
  }
  return true;
}

string to_string(const vector<int64_t> &shape) { return ops::to_string(shape); }

std::string to_string(const ge::Shape &shape) { return to_string(shape.GetDims()); }

std::string to_string(const ge::GeShape &shape) { return to_string(shape.GetDims()); }

std::string to_string(const vector<pair<int64_t, int64_t>> &ranges) { return ops::to_string(ranges); }

bool DynamicShapeInfer::CatchFormatAndShape() {
  inputs = op_desc->GetAllInputName();
  outputs = op_desc->GetAllOutputName();
  GeTensorDescPtr tensor_desc_input, tensor_desc_output;

  // get and save current input shape&format, and assign origin ones to them
  std::string input_name;
  for (map<std::string, uint32_t>::iterator it = inputs.begin(); it != inputs.end(); ++it) {
    input_name = it->first;
    tensor_desc_input = op_desc->MutableInputDesc(input_name);
    if (tensor_desc_input == nullptr) {
      continue;
    }
    Format curr_format = tensor_desc_input->GetFormat();

    map_format.insert(std::pair<std::string, Format>(input_name, curr_format));
    map_dtype.insert(std::pair<std::string, DataType>(input_name, tensor_desc_input->GetDataType()));

    if (tensor_desc_input->GetOriginFormat() == curr_format) {
      continue;
    }
    tensor_desc_input->SetFormat(tensor_desc_input->GetOriginFormat());
    tensor_desc_input->SetShape(tensor_desc_input->GetOriginShape());
  }

  // get and save current output shape&format, and assign origin ones to them
  std::string output_name;
  for (map<std::string, uint32_t>::iterator it = outputs.begin(); it != outputs.end(); ++it) {
    output_name = it->first;
    tensor_desc_output = op_desc->MutableOutputDesc(output_name);
    if (tensor_desc_output == nullptr) {
      continue;
    }
    Format curr_format = tensor_desc_output->GetFormat();

    map_format.insert(std::pair<std::string, Format>(output_name, curr_format));
    map_dtype.insert(std::pair<std::string, DataType>(output_name, tensor_desc_output->GetDataType()));

    if (tensor_desc_output->GetOriginFormat() == curr_format) {
      continue;
    }
    tensor_desc_output->SetFormat(tensor_desc_output->GetOriginFormat());
  }

  return true;
}

bool DynamicShapeInfer::UpdateFormatAndShape() {
  const int64_t opImplType = EN_IMPL_CUSTOM_TBE;
  GeTensorDescPtr tensor_desc_input, tensor_desc_output;
  // assign output's after infershape to origin shape
  for (map<std::string, uint32_t>::iterator it = outputs.begin(); it != outputs.end(); ++it) {
    tensor_desc_output = op_desc->MutableOutputDesc(it->first);
    if (tensor_desc_output == nullptr) {
      continue;
    }
    tensor_desc_output->SetOriginShape(tensor_desc_output->GetShape());
  }

  // transfer input's origin shape to current shape
  Format ori_input_format, cur_input_format;
  GeShape ori_infer_shape, current_shape;
  std::string input_name;
  for (map<std::string, uint32_t>::iterator it = inputs.begin(); it != inputs.end(); ++it) {
    input_name = it->first;
    tensor_desc_input = op_desc->MutableInputDesc(input_name);
    if (tensor_desc_input == nullptr) {
      continue;
    }
    ori_input_format = tensor_desc_input->GetFormat();
    ori_infer_shape = tensor_desc_input->GetShape();
    cur_input_format = map_format[input_name];

    // print some info
    OP_LOGI(TbeGetName(op).c_str(), "origin input shape %s is %s", input_name.c_str(),
            to_string(ori_infer_shape).c_str());

    ShapeAndFormat shapeAndFormatInfoInput = {ori_infer_shape,  current_shape,         ori_input_format,
                                              cur_input_format, map_dtype[input_name], opImplType};
    if (ori_input_format == cur_input_format) {
      // no need to transfer shape
      continue;
    } else {
      ShapeTransferAccordingToFormat *global_object = new (std::nothrow) ShapeTransferAccordingToFormat();
      CHECK(
        global_object == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("new ShapeTransferAccordingToFormat failed.")),
        return false);
      global_object->GetShapeAccordingToFormat(shapeAndFormatInfoInput);

      // print some info
      OP_LOGI(TbeGetName(op).c_str(), "current input shape %s is %s", input_name.c_str(),
              to_string(current_shape).c_str());

      tensor_desc_input->SetFormat(cur_input_format);
      tensor_desc_input->SetShape(current_shape);
      delete global_object;
    }
  }

  // transfer output's origin shape to current shape
  Format ori_output_format, cur_output_format;
  GeShape ori_infer_out_shape, current_out_shape;
  std::string output_name;
  for (map<std::string, uint32_t>::iterator it = outputs.begin(); it != outputs.end(); ++it) {
    output_name = it->first;
    tensor_desc_output = op_desc->MutableOutputDesc(output_name);
    if (tensor_desc_output == nullptr) {
      continue;
    }
    ori_output_format = tensor_desc_output->GetFormat();
    ori_infer_out_shape = tensor_desc_output->GetShape();
    cur_output_format = map_format[output_name];

    // print some info
    OP_LOGI(TbeGetName(op).c_str(), "origin output shape %s is %s", output_name.c_str(),
            to_string(ori_infer_out_shape).c_str());

    ShapeAndFormat shapeAndFormatInfoOutput = {ori_infer_out_shape, current_out_shape,      ori_output_format,
                                               cur_output_format,   map_dtype[output_name], opImplType};
    if (ori_output_format == cur_output_format) {
      // no need to transfer shape
      continue;
    } else {
      ShapeTransferAccordingToFormat *global_object = new (std::nothrow) ShapeTransferAccordingToFormat();
      CHECK(
        global_object == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("new ShapeTransferAccordingToFormat failed.")),
        return false);
      global_object->GetShapeAccordingToFormat(shapeAndFormatInfoOutput);

      // print some info
      OP_LOGI(TbeGetName(op).c_str(), "current output shape %s is %s", output_name.c_str(),
              to_string(current_out_shape).c_str());

      tensor_desc_output->SetFormat(cur_output_format);
      tensor_desc_output->SetShape(current_out_shape);
      delete global_object;
    }
  }

  return true;
}

bool IsEmptyTensor(const std::vector<int64_t> &dims) {
  if (dims.size() == 1 && dims[0] == 0) {
    return true;
  } else {
    return false;
  }
}

bool IsUnknownRank(const Operator &op, const std::string &tensor_name, const std::string &types) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_desc == nullptr, VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("invalid OpDesc.")),
        return false);
  GeTensorDescPtr tensor_desc;
  if (types == "input") {
    tensor_desc = op_desc->MutableInputDesc(tensor_name);
  } else if (types == "output") {
    tensor_desc = op_desc->MutableOutputDesc(tensor_name);
  } else {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                        OtherErrMsg(ConcatString("invalid params:", types, " of types to judge.")));
    return false;
  }

  std::vector<int64_t> shape_vec = tensor_desc->GetShape().GetDims();
  if (shape_vec.size() == 1 && shape_vec[0] == INPUT_NEGATIVE_NUM2) {
    return true;
  }
  return false;
}

bool IsUnknownRankShape(const std::vector<int64_t> &shape_vec) {
  if (shape_vec.size() == 1 && shape_vec[0] == -2) {
    return true;
  }
  return false;
}

bool IsUnknownRankShape(const GeShape &input_shape) { return input_shape.IsUnknownDimNum(); }

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

void MakeUpShapeRange(const ge::GeShape &shape, std::vector<std::pair<int64_t, int64_t>> &range) {
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

bool OneInOneOutDynamicInfer(const Operator &op, const std::string &input_name,
                             const std::vector<std::string> &output_name_list) {
  // get input desc
  PROFILING_PROTO_INIT(TbeGetName(op).c_str());
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_info == nullptr, VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("invalid OpDesc.")),
        return false);
  auto input_desc = op_info->MutableInputDesc(input_name);
  CHECK(input_desc == nullptr, VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("invalid input_desc")),
        return false);
  vector<int64_t> input_shape = input_desc->MutableShape().GetDims();
  DataType input_dtype = input_desc->GetDataType();

  if (IsUnknown(input_shape)) {
    std::vector<std::pair<int64_t, int64_t>> input_range;
    input_desc->GetShapeRange(input_range);
    MakeUpShapeRange(input_shape, input_range);

    auto output_desc = op_info->MutableOutputDesc(0);
    CHECK(output_desc == nullptr,
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("invalid output_desc")), return false);
    for (const string &output_name : output_name_list) {
      output_desc = op_info->MutableOutputDesc(output_name);
      output_desc->SetShape(GeShape(input_shape));
      output_desc->SetOriginShape(GeShape(input_shape));
      output_desc->SetShapeRange(input_range);
      output_desc->SetDataType(input_dtype);
    }
  } else {
    auto output_desc = op_info->MutableOutputDesc(0);
    CHECK(output_desc == nullptr,
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("invalid output_desc")), return false);
    PROFILING_PROTO_AFTER_GET_SHAPE_REG();
    PROFILING_PROTO_AFTER_INFER_SHAPE_REG();
    for (const string &output_name : output_name_list) {
      output_desc = op_info->MutableOutputDesc(output_name);
      output_desc->SetShape(GeShape(input_shape));
      output_desc->SetDataType(input_dtype);
    }
    PROFILING_PROTO_END();
  }
  return true;
}

bool OneInOneOutDynamicInfer(const Operator &op, const int64_t &input_idx,
                             const std::vector<int64_t> &output_idx_list) {
  // get input desc
  PROFILING_PROTO_INIT(TbeGetName(op).c_str());
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_info == nullptr, VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("invalid OpDesc.")),
        return false);
  auto input_desc = op_info->MutableInputDesc(input_idx);
  CHECK(input_desc == nullptr, VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("invalid input_desc")),
        return false);
  const GeShape &input_shape = input_desc->MutableShape();
  DataType input_dtype = input_desc->GetDataType();

  if (input_shape.IsUnknownShape()) {
    std::vector<std::pair<int64_t, int64_t>> input_range;
    input_desc->GetShapeRange(input_range);
    std::vector<int64_t> input_shape_vec = input_shape.GetDims();
    MakeUpShapeRange(input_shape_vec, input_range);

    for (const int64_t &output_idx : output_idx_list) {
      auto output_desc = op_info->MutableOutputDesc(output_idx);
      CHECK(output_desc == nullptr,
            VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("invalid output_desc")), return false);
      output_desc->SetShape(input_shape);
      output_desc->SetShapeRange(input_range);
      output_desc->SetDataType(input_dtype);
    }
  } else {
    PROFILING_PROTO_AFTER_GET_SHAPE_REG();
    PROFILING_PROTO_AFTER_INFER_SHAPE_REG();
    for (const int64_t &output_idx : output_idx_list) {
      auto output_desc = op_info->MutableOutputDesc(output_idx);
      CHECK(output_desc == nullptr,
            VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("invalid output_desc")), return false);
      output_desc->SetShape(input_shape);
      output_desc->SetDataType(input_dtype);
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
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_info == nullptr || op_info->MutableInputDesc(input1_name) == nullptr ||
          op_info->MutableInputDesc(input2_name) == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("invalid OpDesc.")), return false);
  auto input1_desc = op_info->MutableInputDesc(input1_name);
  vector<int64_t> input1_shape = input1_desc->MutableShape().GetDims();
  DataType input_dtype = input1_desc->GetDataType();

  // get input2 desc
  auto input2_desc = op_info->MutableInputDesc(input2_name);
  vector<int64_t> input2_shape = input2_desc->MutableShape().GetDims();

  if (IsUnknown(input1_shape) || IsUnknown(input2_shape)) {
    std::vector<std::pair<int64_t, int64_t>> input1_range;
    input1_desc->GetShapeRange(input1_range);
    std::vector<std::pair<int64_t, int64_t>> input2_range;
    input2_desc->GetShapeRange(input2_range);

    vector<int64_t> dim_size = {};
    FixShapeRangeWithDims(dim_size, input1_shape, input2_shape, input1_range, input2_range);

    // update output desc
    for (const string &output_name : output_name_list) {
      auto output_desc = op_info->MutableOutputDesc(output_name);
      output_desc->SetShape(GeShape(input1_shape));
      output_desc->SetOriginShape(GeShape(input1_shape));
      output_desc->SetShapeRange(input1_range);
      output_desc->SetDataType(input_dtype);
    }
  } else {
    for (const string &output_name : output_name_list) {
      auto output_desc = op_info->MutableOutputDesc(output_name);
      output_desc->SetShape(GeShape(input1_shape));
      output_desc->SetDataType(input_dtype);
    }
  }
  return true;
}

bool SetScalarOutputDesc(const string &input, const string &output, OpDescPtr op_desc, GeShape &output_shape) {
  if (output_shape.IsScalar()) {
    auto td = op_desc->MutableOutputDesc(output);
    td->SetShape(output_shape);
    td->SetOriginShape(output_shape);
    td->SetDataType(op_desc->MutableInputDesc(input)->GetDataType());
    td->SetOriginDataType(op_desc->MutableInputDesc(input)->GetDataType());
    return true;
  } else {
    return false;
  }
}

bool IsEmptyTensor(GeTensorDescPtr tensor_desc) { return IsEmptyTensor(tensor_desc->MutableShape()); }

bool IsEmptyTensor(const GeShape &ge_shape) {
  bool is_empty = false;
  for (const auto &dim : ge_shape.GetDims()) {
    if (dim == 0) {
      is_empty = true;
      break;
    }
  }
  return is_empty;
}

std::string RangeToString(const std::vector<std::pair<int64_t, int64_t>> &ranges) {
  bool first = true;
  std::stringstream ss;
  ss << "[";
  for (const auto &range : ranges) {
    if (first) {
      first = false;
    } else {
      ss << ",";
    }
    ss << "{";
    ss << range.first << "," << range.second;
    ss << "}";
  }
  ss << "]";
  return ss.str();
}

std::string AxisTypeInfoToString(const std::vector<ge::AxisTypeInfo> &axis_type_infos) {
  if (axis_type_infos.empty()) {
    return "[]";
  }

  std::set<std::string> infos;
  for (ge::AxisTypeInfo axis_type_info : axis_type_infos) {
    std::string info{"{type:"};
    info += std::to_string(static_cast<int>(axis_type_info.GetAxisType()));
    info += ",relate_inputs:[";

    std::vector<ge::CutInfo> inputs = axis_type_info.GetRelateInputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (i > 0) {
        info += ",";
      }
      info += "{";

      info += std::to_string(inputs[i].first);
      info += ",{";
      std::vector<int64_t> axes = inputs[i].second;
      if (!axes.empty()) {
        info += std::accumulate(std::next(axes.begin()), axes.end(), std::to_string(*axes.begin()),
                                [](const std::string &a, int64_t b) { return a + "," + std::to_string(b); });
      }
      info += "}}";
    }
    info += "],relate_outputs:[";

    std::vector<ge::CutInfo> outputs = axis_type_info.GetRelateOutputs();
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (i > 0) {
        info += ",";
      }
      info += "{";

      info += std::to_string(outputs[i].first);
      info += ",{";
      std::vector<int64_t> axes = outputs[i].second;
      if (!axes.empty()) {
        info += std::accumulate(std::next(axes.begin()), axes.end(), std::to_string(*axes.begin()),
                                [](const std::string &a, int64_t b) { return a + "," + std::to_string(b); });
      }
      info += "}}";
    }

    info += "]}";
    (void)infos.insert(info);
  }

  return "[" +
         std::accumulate(std::next(infos.begin()), infos.end(), *infos.begin(),
                         [](const std::string &a, const std::string &b) { return a + "," + b; }) +
         "]";
}

ge::graphStatus GetAllValidInputsWithIndices(const Operator &op, map<uint32_t, ConstGeTensorDescPtr> &inputs) {
  OP_LOGD(TbeGetName(op), "Get all valid inputs desc with indices begin");

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_desc == nullptr, INFER_AXIS_TYPE_ERR_REPORT(TbeGetName(op), "Failed to get op desc"), return GRAPH_FAILED);

  uint32_t inputs_size = static_cast<uint32_t>(op_desc->GetAllInputsSize());
  for (uint32_t i = 0; i < inputs_size; ++i) {
    ConstGeTensorDescPtr input_desc_ptr = op_desc->GetInputDescPtr(i);
    if (input_desc_ptr) {
      (void)inputs.emplace(i, input_desc_ptr);
    } else if (op_desc->IsOptionalInput(i)) {
      OP_LOGD(TbeGetName(op), "Failed to get optional input desc i(%u). Maybe it is not set up", i);
    } else {
      INFER_AXIS_TYPE_ERR_REPORT(TbeGetName(op), "Failed to get input desc i(%u)", i);
      return GRAPH_FAILED;
    }
  }

  OP_LOGD(TbeGetName(op), "Get all valid inputs desc with indices end");
  return GRAPH_SUCCESS;
}

ge::graphStatus GetAllValidOutputsWithIndices(const Operator &op, map<uint32_t, ConstGeTensorDescPtr> &outputs) {
  OP_LOGD(TbeGetName(op), "Get all valid outputs desc with indices begin");

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_desc == nullptr, INFER_AXIS_TYPE_ERR_REPORT(TbeGetName(op), "Failed to get op desc"), return GRAPH_FAILED);

  uint32_t outputs_size = op_desc->GetAllOutputsDescSize();
  for (uint32_t i = 0; i < outputs_size; ++i) {
    ConstGeTensorDescPtr output_desc_ptr = op_desc->GetOutputDescPtr(i);
    if (output_desc_ptr) {
      (void)outputs.emplace(i, output_desc_ptr);
    } else {
      INFER_AXIS_TYPE_ERR_REPORT(TbeGetName(op), "Failed to get output desc i(%u)", i);
      return GRAPH_FAILED;
    }
  }

  OP_LOGD(TbeGetName(op), "Get all valid outputs desc with indices end");
  return GRAPH_SUCCESS;
}

ge::graphStatus GetShapesForInferAxisType(const Operator &op, map<uint32_t, vector<int64_t>> &input_shapes,
                                          map<uint32_t, vector<int64_t>> &output_shapes, bool &is_unknown_dim_num,
                                          size_t &dim_num) {
  OP_LOGD(TbeGetName(op), "Get tensor shapes for infer axis type begin");

  map<uint32_t, ConstGeTensorDescPtr> outputs;
  CHECK(GetAllValidOutputsWithIndices(op, outputs) != GRAPH_SUCCESS,
        INFER_AXIS_TYPE_ERR_REPORT(TbeGetName(op), "Failed to get valid outputs desc"), return GRAPH_FAILED);
  for (auto iter : outputs) {
    GeShape shape = iter.second->GetShape();
    if (shape.IsUnknownDimNum()) {
      is_unknown_dim_num = true;
      OP_LOGD(TbeGetName(op), "Input(%u) has unknown rank shape", iter.first);
      return GRAPH_SUCCESS;
    }
    if (!shape.IsScalar()) {
      (void)output_shapes.emplace(iter.first, shape.GetDims());
      dim_num = std::max(dim_num, shape.GetDimNum());
    }
  }
  CHECK(dim_num == 0, OP_LOGD(TbeGetName(op), "Unsupported for all tensors are scalar"), return GRAPH_SUCCESS);

  map<uint32_t, ConstGeTensorDescPtr> inputs;
  CHECK(GetAllValidInputsWithIndices(op, inputs) != GRAPH_SUCCESS,
        INFER_AXIS_TYPE_ERR_REPORT(TbeGetName(op), "Failed to get valid inputs desc"), return GRAPH_FAILED);
  for (auto iter : inputs) {
    GeShape shape = iter.second->GetShape();
    if (shape.IsUnknownDimNum()) {
      is_unknown_dim_num = true;
      OP_LOGD(TbeGetName(op), "Output(%u) has unknown rank shape", iter.first);
      return GRAPH_SUCCESS;
    }
    if (!shape.IsScalar()) {
      (void)input_shapes.emplace(iter.first, shape.GetDims());
      dim_num = std::max(dim_num, shape.GetDimNum());
    }
  }

  OP_LOGD(TbeGetName(op), "Get tensor shapes for infer axis type end");
  return GRAPH_SUCCESS;
}

ge::graphStatus GetDimNumForOp(const Operator &op, const vector<int64_t> &input_ids, const vector<int64_t> &output_ids,
                               size_t &dim_num) {
  OP_LOGD(TbeGetName(op), "Get inputs/outputs dim num for Operator.");
  dim_num = 0;
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_desc == nullptr, OP_LOGE(TbeGetName(op), "Failed to get desc from operator. Please check the node info."),
        return GRAPH_FAILED);

  for (size_t idx = 0; idx < input_ids.size(); ++idx) {
    ge::GeShape shape = op_desc->GetInputDescPtr(input_ids[idx])->GetShape();
    if (shape.IsUnknownDimNum()) {
      dim_num = ge::UNKNOWN_DIM_NUM;
      OP_LOGD(TbeGetName(op), "Input(%zu) has unknown rank shape", idx);
      return GRAPH_SUCCESS;
    } else {
      dim_num = std::max(dim_num, shape.GetDimNum());
    }
  }

  for (size_t idx = 0; idx < output_ids.size(); ++idx) {
    ge::GeShape shape = op_desc->GetOutputDescPtr(output_ids[idx])->GetShape();
    if (shape.IsUnknownDimNum()) {
      dim_num = ge::UNKNOWN_DIM_NUM;
      OP_LOGD(TbeGetName(op), "Output(%zu) has unknown rank shape", idx);
      return GRAPH_SUCCESS;
    } else {
      dim_num = std::max(dim_num, shape.GetDimNum());
    }
  }

  OP_LOGD(TbeGetName(op), "Get inputs/outputs dim num for Operator.");
  return GRAPH_SUCCESS;
}

ge::graphStatus InferAxisType4ElementwiseOp(const Operator &op, vector<AxisTypeInfo> &axis_type) {
  OP_LOGD(TbeGetName(op), "Infer axis type for %s begin.", TbeGetOpType(op).c_str());

  map<uint32_t, vector<int64_t>> input_shapes;
  map<uint32_t, vector<int64_t>> output_shapes;
  bool is_unknown_dim_num = false;
  size_t dim_num = 0;
  CHECK(GetShapesForInferAxisType(op, input_shapes, output_shapes, is_unknown_dim_num, dim_num) != GRAPH_SUCCESS,
        INFER_AXIS_TYPE_ERR_REPORT(TbeGetName(op), "Failed to get output shapes"), return GRAPH_FAILED);
  CHECK(is_unknown_dim_num, OP_LOGD(TbeGetName(op), "Unsupported for unknown rank"), return GRAPH_SUCCESS);
  CHECK(dim_num == 0, OP_LOGD(TbeGetName(op), "Unsupported for all input/output are scalar"), return GRAPH_SUCCESS);

  for (auto iter : input_shapes) {
    CHECK(dim_num != iter.second.size(),
          INFER_AXIS_TYPE_ERR_REPORT(TbeGetName(op),
                                     "Failed to check shape rank, input %u dim num is %zu, not equals rank %zu",
                                     iter.first, iter.second.size(), dim_num),
          return GRAPH_FAILED);
  }
  for (auto iter : output_shapes) {
    CHECK(dim_num != iter.second.size(),
          INFER_AXIS_TYPE_ERR_REPORT(TbeGetName(op),
                                     "Failed to check shape rank, output %u dim num is %zu, not equals rank %zu",
                                     iter.first, iter.second.size(), dim_num),
          return GRAPH_FAILED);
  }

  for (size_t axis = 0; axis < dim_num; ++axis) {
    AxisTypeInfoBuilder builder = AxisTypeInfoBuilder().AxisType(AxisType::ELEMENTWISE);

    for (auto iter : output_shapes) {
      if (iter.second[axis] != 1) {
        (void)builder.AddOutputCutInfo({static_cast<int64_t>(iter.first), {static_cast<int64_t>(axis)}});
      }
    }

    if (!builder.IsRelateOutputsEmpty()) {
      for (auto iter : input_shapes) {
        if (iter.second[axis] != 1) {
          (void)builder.AddInputCutInfo({static_cast<int64_t>(iter.first), {static_cast<int64_t>(axis)}});
        }
      }
      axis_type.push_back(builder.Build());
    }
  }

  OP_LOGD(TbeGetName(op), "InferAxisType for element-wise op end. AxisTypeInfo is: %s",
          AxisTypeInfoToString(axis_type).c_str());
  return GRAPH_SUCCESS;
}

ge::graphStatus InferAxisType4BroadcastOp(const Operator &op, vector<AxisTypeInfo> &axis_type) {
  OP_LOGD(TbeGetName(op), "Infer axis type for broadcast op begin");

  map<uint32_t, vector<int64_t>> input_shapes;
  map<uint32_t, vector<int64_t>> output_shapes;
  bool is_unknown_dim_num = false;
  size_t dim_num = 0;
  CHECK(GetShapesForInferAxisType(op, input_shapes, output_shapes, is_unknown_dim_num, dim_num) != GRAPH_SUCCESS,
        INFER_AXIS_TYPE_ERR_REPORT(TbeGetName(op), "Failed to get output shapes"), return GRAPH_FAILED);
  CHECK(is_unknown_dim_num, OP_LOGD(TbeGetName(op), "Unsupported for unknown rank"), return GRAPH_SUCCESS);
  CHECK(dim_num == 0, OP_LOGD(TbeGetName(op), "Unsupported for all input/output are scalar"), return GRAPH_SUCCESS);

  for (size_t axis = 0; axis < dim_num; ++axis) {
    AxisTypeInfoBuilder builder = AxisTypeInfoBuilder().AxisType(AxisType::ELEMENTWISE);

    for (auto iter : output_shapes) {
      if (iter.second[axis] != 1) {
        (void)builder.AddOutputCutInfo({static_cast<int64_t>(iter.first), {static_cast<int64_t>(axis)}});
      }
    }

    if (!builder.IsRelateOutputsEmpty()) {
      for (auto iter : input_shapes) {
        // make the dimension right-aligned
        size_t fix_axis = iter.second.size() + axis - dim_num;
        if (iter.second.size() + axis >= dim_num && iter.second[fix_axis] != 1) {
          (void)builder.AddInputCutInfo({static_cast<int64_t>(iter.first), {static_cast<int64_t>(fix_axis)}});
        }
      }
      axis_type.push_back(builder.Build());
    }
  }

  OP_LOGD(TbeGetName(op), "InferAxisType for broadcast op end. AxisTypeInfo is: %s",
          AxisTypeInfoToString(axis_type).c_str());
  return GRAPH_SUCCESS;
}

ge::graphStatus InferElementwiseAxisTypeHelper(const Operator &op, vector<AxisTypeInfo> &axis_type,
                                               const vector<int64_t> &allowed_split_inputs,
                                               const vector<int64_t> &allowed_split_outputs,
                                               vector<int64_t> &excepted_axes) {
  OP_LOGD(TbeGetName(op), "Infer elementwise axis type for op begin");

  size_t dim_num = 0;
  CHECK(GetDimNumForOp(op, allowed_split_inputs, allowed_split_outputs, dim_num) != GRAPH_SUCCESS,
        INFER_AXIS_TYPE_ERR_REPORT(TbeGetName(op), "Infer axis type, failed to get input/output shapes."),
        return GRAPH_FAILED);
  CHECK(dim_num == static_cast<size_t>(ge::UNKNOWN_DIM_NUM),
        OP_LOGD(TbeGetName(op), "No axis can be cut for shape is unknown rank. Can not infer axis type."),
        return GRAPH_SUCCESS);
  CHECK(dim_num == 0, OP_LOGD(TbeGetName(op), "Unsupported for all input/output are scalar. Can not infer axis type."),
        return GRAPH_SUCCESS);

  for (size_t idx = 0; idx < excepted_axes.size(); ++idx) {
    if (excepted_axes[idx] < 0) {
      excepted_axes[idx] = excepted_axes[idx] + dim_num;
    }
  }

  for (size_t axis = 0; axis < dim_num; ++axis) {
    AxisTypeInfoBuilder builder = AxisTypeInfoBuilder().AxisType(AxisType::ELEMENTWISE);
    auto axis_it = std::find(excepted_axes.begin(), excepted_axes.end(), axis);
    if (axis_it == excepted_axes.end()) {
      for (size_t idx = 0; idx < allowed_split_outputs.size(); ++idx) {
        builder.AddOutputCutInfo({allowed_split_outputs[idx], {static_cast<int64_t>(axis)}});
      }
      for (size_t idx = 0; idx < allowed_split_inputs.size(); ++idx) {
        builder.AddInputCutInfo({allowed_split_inputs[idx], {static_cast<int64_t>(axis)}});
      }
      axis_type.push_back(builder.Build());
    }
  }

  OP_LOGD(TbeGetName(op), "Infer elementwise axis type for op end. AxisTypeInfo is: %s",
          AxisTypeInfoToString(axis_type).c_str());
  return GRAPH_SUCCESS;
}

ge::graphStatus InferAxisType4ReduceOpHelper(const Operator &op, const AxisType &reduce_type,
                                             const std::vector<int64_t> &axis, const bool &keep_dims,
                                             vector<AxisTypeInfo> &axis_type) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_desc == nullptr,
        INFER_AXIS_TYPE_ERR_REPORT(TbeGetName(op), "Failed to get desc from operator. Please check the node info."),
        return GRAPH_FAILED);

  GeShape x_shape = op_desc->GetInputDescPtr(0)->GetShape();
  CHECK(x_shape.IsUnknownDimNum(), OP_LOGD(TbeGetName(op), "No axis can be cut for input shape is unknown rank."),
        return GRAPH_SUCCESS);
  CHECK(x_shape.IsScalar(), OP_LOGD(TbeGetName(op), "No axis can be cut for input_x is scalar"), return GRAPH_SUCCESS);

  GeShape y_shape = op_desc->GetOutputDescPtr(0)->GetShape();
  CHECK(y_shape.IsUnknownDimNum(), OP_LOGD(TbeGetName(op), "No axis can be cut for output shape is unknown rank."),
        return GRAPH_SUCCESS);

  std::vector<int64_t> x_dims = x_shape.GetDims();
  int64_t rank = static_cast<int64_t>(x_dims.size());
  OP_LOGD(TbeGetName(op), "output shape = %s, axis = %s", to_string(y_shape).c_str(), to_string(axis).c_str());

  int64_t out_index = 0;
  for (int64_t dim = 0; dim < rank; dim++) {
    if (find(axis.begin(), axis.end(), dim) != axis.end()) {
      if (x_dims[dim] != 1) {
        AxisTypeInfoBuilder builder =
          AxisTypeInfoBuilder().AxisType(reduce_type).AddInputCutInfo({0, {dim}}).AddOutputCutInfo({0, {}});
        axis_type.emplace_back(builder.Build());
      }
      if (keep_dims) {
        out_index = out_index + 1;
      }
    } else {
      if (x_dims[dim] != 1) {
        AxisTypeInfoBuilder builder = AxisTypeInfoBuilder()
                                        .AxisType(AxisType::ELEMENTWISE)
                                        .AddInputCutInfo({0, {dim}})
                                        .AddOutputCutInfo({0, {out_index}});

        axis_type.emplace_back(builder.Build());
      }
      out_index = out_index + 1;
    }
  }

  OP_LOGD(TbeGetName(op), "InferAxisType for reduce op end. AxisTypeInfo is: %s",
          AxisTypeInfoToString(axis_type).c_str());
  return GRAPH_SUCCESS;
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
                             const GeShape &x_shape) {
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
                                       std::vector<std::pair<int64_t, int64_t>> &y_shape_range, GeShape &y_shape) {
  y_shape = GeShape(std::vector<int64_t>(y_rank, UNKNOWN_DIM));
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
                                   std::vector<std::pair<int64_t, int64_t>> &y_shape_range, GeShape &y_shape) {
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
  y_shape = GeShape(y_dims);

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

void ReshapeRangeInferAllDims(const Operator &op, const std::vector<std::pair<int64_t, int64_t>> &x_shape_range,
                              const GeShape &x_shape, const std::vector<std::pair<int64_t, int64_t>> &shape_value_range,
                              int64_t y_rank, std::vector<std::pair<int64_t, int64_t>> &y_shape_range,
                              GeShape &y_shape) {
  // step 1, calculate input_x range max and init worst y shape and y shape range
  int64_t max_elements_count = CalcMaxElementsCount(op, x_shape_range, x_shape);
  GenerateWorstYShapeAndYShapeRange(y_rank, max_elements_count, y_shape_range, y_shape);
  if (shape_value_range.empty()) {
    // no value range, can not calculate accurate shape range.
    return;
  }

  // step 2, deal with empty tensor. if no value range cannot infer empty tensor.
  std::vector<std::pair<int64_t, int64_t>> value_range = shape_value_range;
  bool has_zero_in_range = RepairAndCheckRange(x_shape_range, value_range);
  if (IsEmptyTensor(x_shape) || has_zero_in_range) {
    InferShapeRangeForEmptyTensor(y_rank, max_elements_count, value_range, y_shape_range, y_shape);
    return;
  }

  // step 3, calculate accurate dims and shape_range
  std::vector<int64_t> y_dims = y_shape.GetDims();
  UpdateDimsAndShapeRange(op, max_elements_count, value_range, y_dims, y_shape_range);
  y_shape = GeShape(y_dims);
}

void FixRangeMaxToInt32max(GeShape &shape, std::vector<std::pair<int64_t, int64_t>> &shape_range) {
  int64_t int32_max = static_cast<int64_t>(INT32_MAX);
  auto dims = shape.GetDims();
  for (size_t i = 0; i < dims.size(); ++i) {
    shape_range[i].first = std::min(shape_range[i].first, int32_max);
    shape_range[i].second = std::min(shape_range[i].second, int32_max);
    dims[i] = std::min(dims[i], int32_max);
  }
  shape = GeShape(dims);
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

void ReshapeRangeInfer(const Operator &op, const std::vector<std::pair<int64_t, int64_t>> &x_range,
                       std::vector<std::pair<int64_t, int64_t>> &y_range, GeShape &output_shape) {
  int64_t max_input_dims = CalculateMaxInputDims(x_range, op);
  if (max_input_dims < 0) {
    for (const auto dim : output_shape.GetDims()) {
      if (dim < 0) {
        y_range.emplace_back(std::pair<int64_t, int64_t>(0, -1));
      } else {
        y_range.emplace_back(std::pair<int64_t, int64_t>(dim, dim));
      }
    }
  } else {
    std::vector<size_t> unknown_dim_indexes;
    size_t dim_index = 0U;
    int64_t left = max_input_dims;
    left = (left > INT32_MAX) ? INT32_MAX : left;
    for (const auto dim : output_shape.GetDims()) {
      if (dim < 0) {
        y_range.emplace_back(std::pair<int64_t, int64_t>(1, left));
        unknown_dim_indexes.emplace_back(dim_index);
      } else {
        y_range.emplace_back(std::pair<int64_t, int64_t>(dim, dim));
        if (dim != 0) {
          left = static_cast<int64_t>((static_cast<double>(left) + HALF) / dim);
        }
      }
      dim_index++;
    }

    if (unknown_dim_indexes.size() == 1U) {
      y_range[unknown_dim_indexes[0U]] = std::pair<int64_t, int64_t>(1, left);
    }
  }
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
}  // namespace ge
