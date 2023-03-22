/**
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file add_dsl.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "./add_dsl.h"
#include <string>
#include <vector>

namespace ge {

bool InferShapeAndTypeAdd(Operator &op, const string &input_name1, const string &input_name2,
                          const string &output_name) {
  TensorDesc vOutputDesc = op.GetOutputDescByName(output_name.c_str());

  DataType input_dtype = op.GetInputDescByName(input_name1.c_str()).GetDataType();
  Format input_format = op.GetInputDescByName(input_name1.c_str()).GetFormat();
  // 针对shape维度大小进行交换
  ge::Shape shapeX = op.GetInputDescByName(input_name1.c_str()).GetShape();
  ge::Shape shapeY = op.GetInputDescByName(input_name2.c_str()).GetShape();
  std::vector<int64_t> dimsX = shapeX.GetDims();
  std::vector<int64_t> dimsY = shapeY.GetDims();
  if (dimsX.size() < dimsY.size()) {
    std::vector<int64_t> dimsTmp = dimsX;
    dimsX = dimsY;
    dimsY = dimsTmp;
  }

  // 对小的shape进行1补齐
  if (dimsX.size() != dimsY.size()) {
    int dec = dimsX.size() - dimsY.size();
    for (int i = 0; i < dec; i++) {
      dimsY.insert(dimsY.begin(), (int64_t)1);
    }
  }

  // 设置输出的shape维度
  std::vector<int64_t> dimVec;
  for (size_t i = 0; i < dimsX.size(); i++) {
    if ((dimsX[i] != dimsY[i]) && (dimsX[i] != 1) && (dimsY[i] != 1)) {
      return false;
    }

    int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
    dimVec.push_back(dims);
  }
  ge::Shape outputShape = ge::Shape(dimVec);

  vOutputDesc.SetShape(outputShape);
  vOutputDesc.SetDataType(input_dtype);
  vOutputDesc.SetFormat(input_format);
  op.UpdateOutputDesc(output_name.c_str(), vOutputDesc);

  return true;
}

//----------------Add-------------------
IMPLEMT_VERIFIER(AddDsl, AddVerify) {
  if (op.GetInputDescByName("x1").GetDataType() != op.GetInputDescByName("x2").GetDataType()) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(AddInferShape) {
  if (InferShapeAndTypeAdd(op, "x1", "x2", "y")) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(AddDsl, AddInferShape);

// Registered verify function
VERIFY_FUNC_REG(AddDsl, AddVerify);
//----------------Add-------------------
}  // namespace ge
