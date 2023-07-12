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
 * \file transfer_shape_according_to_format.cpp
 * \brief set shape according to original format and current format
 */
#include "transfer_shape_according_to_format.h"
#include <memory>
#include "framework/omg/omg_inner_types.h"

namespace ge {
ShapeTransferAccordingToFormat::ShapeTransferAccordingToFormat()
    : getNewShapeFuncMap(
        {{ge::FORMAT_NCHW, std::make_shared<GetNewShapeByAxisValueAndFormat>(GetNCHWShapeByAxisValue)},
         {ge::FORMAT_NHWC, std::make_shared<GetNewShapeByAxisValueAndFormat>(GetNHWCShapeByAxisValue)},
         {ge::FORMAT_NC1HWC0, std::make_shared<GetNewShapeByAxisValueAndFormat>(GetNC1HWC0ShapeByAxisValue)},
         {ge::FORMAT_FRACTAL_Z, std::make_shared<GetNewShapeByAxisValueAndFormat>(GetFzShapeByAxisValue)},
         {ge::FORMAT_HWCN, std::make_shared<GetNewShapeByAxisValueAndFormat>(GetHWCNShapeByAxisValue)},
         {ge::FORMAT_C1HWNCoC0, std::make_shared<GetNewShapeByAxisValueAndFormat>(GetC1HWNCoC0ShapeByAxisValue)},
         {ge::FORMAT_FRACTAL_NZ, std::make_shared<GetNewShapeByAxisValueAndFormat>(GetNzShapeByAxisValue)}}),

      mapOfDtypeAndC0({{ge::DT_FLOAT16, SHAPE_NUMBER_16},
                       {ge::DT_FLOAT, SHAPE_NUMBER_16},
                       {ge::DT_INT8, SHAPE_NUMBER_32},
                       {ge::DT_INT16, SHAPE_NUMBER_16},
                       {ge::DT_INT32, SHAPE_NUMBER_16},
                       {ge::DT_INT64, SHAPE_NUMBER_16},
                       {ge::DT_UINT8, SHAPE_NUMBER_16},
                       {ge::DT_UINT16, SHAPE_NUMBER_32},
                       {ge::DT_UINT32, SHAPE_NUMBER_16},
                       {ge::DT_UINT64, SHAPE_NUMBER_16},
                       {ge::DT_BOOL, SHAPE_NUMBER_16}}) {}

bool ShapeTransferAccordingToFormat::GetNCHWShapeByAxisValue(ge::GeShape &newShape, const int64_t &implType,
                                                             const vector<int64_t> &axisValue,
                                                             const vector<int64_t> &ndValue) {
  CHECK(axisValue.size() <= static_cast<size_t>(AXIS_W), LOG_INFO("AxisValue is not correct!"), return true);
  /* axisValue is initialized as a size 6 vector. */
  std::vector<int64_t> newDimVec;
  newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_N]));
  newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_C]));
  newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_H]));
  newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_W]));
  newShape = ge::GeShape(newDimVec);
  return true;
}

bool ShapeTransferAccordingToFormat::GetNHWCShapeByAxisValue(ge::GeShape &newShape, const int64_t &implType,
                                                             const vector<int64_t> &axisValue,
                                                             const vector<int64_t> &ndValue) {
  CHECK(axisValue.size() <= static_cast<size_t>(AXIS_W), LOG_INFO("AxisValue is not correct!"), return true);
  /* axisValue is initialized as a size 6 vector. */
  std::vector<int64_t> newDimVec;
  newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_N]));
  newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_H]));
  newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_W]));
  newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_C]));
  newShape = ge::GeShape(newDimVec);
  return true;
}

bool ShapeTransferAccordingToFormat::GetNC1HWC0ShapeByAxisValue(ge::GeShape &newShape, const int64_t &implType,
                                                                const vector<int64_t> &axisValue,
                                                                const vector<int64_t> &ndValue) {
  CHECK(axisValue.empty(), LOG_INFO("AxisValue is empty!"), return true);
  /* axisValue is initialized as a size 6 vector. */
  std::vector<int64_t> newDimVec;
  if (implType == static_cast<int64_t>(EN_IMPL_HW_TBE) || implType == static_cast<int64_t>(EN_IMPL_CUSTOM_TBE) ||
      implType == static_cast<int64_t>(EN_IMPL_NON_PERSISTENT_CUSTOM_TBE)) {
    CHECK(axisValue.size() <= static_cast<size_t>(AXIS_C0), LOG_INFO("AxisValue is not correct!"), return true);
    newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_N]));
    newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_C1]));
    newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_H]));
    newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_W]));
    newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_C0]));
    newShape = ge::GeShape(newDimVec);
  } else {
    CHECK(axisValue.size() <= static_cast<size_t>(AXIS_W), LOG_INFO("AxisValue is not correct!"), return true);
    newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_N]));
    newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_C]));
    newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_H]));
    newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_W]));
    newShape = ge::GeShape(newDimVec);
  }
  return true;
}

bool ShapeTransferAccordingToFormat::GetFzShapeByAxisValue(ge::GeShape &newShape, const int64_t &implType,
                                                           const vector<int64_t> &axisValue,
                                                           const vector<int64_t> &ndValue) {
  CHECK(axisValue.empty(), LOG_INFO("AxisValue is empty!"), return true);
  /* axisValue is initialized as a size 6 vector. */
  std::vector<int64_t> newDimVec;

  if (ndValue.size() == SIZE_OF_CN) {
    CHECK(axisValue.size() <= static_cast<size_t>(AXIS_C0), LOG_INFO("AxisValue is not correct!"), return true);
    auto sizeOfOriginalVec = ndValue.size();
    newDimVec = ndValue;
    /* sizeOfOriginalVec - 1 mean the last value of original vec
     * sizeOfOriginalVec - 2 mean the second last value of original vec */
    newDimVec[sizeOfOriginalVec - MINUS_VALUE_ONE] =
      DivisionCeiling(ndValue[sizeOfOriginalVec - MINUS_VALUE_ONE], SHAPE_NUMBER_16);
    newDimVec[sizeOfOriginalVec - MINUS_VALUE_TWO] =
      DivisionCeiling(ndValue[sizeOfOriginalVec - MINUS_VALUE_TWO], static_cast<size_t>(axisValue[AXIS_C0]));
    newDimVec.push_back(SHAPE_NUMBER_16);
    newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_C0]));
    newShape = ge::GeShape(newDimVec);
  } else {
    if (implType == static_cast<int64_t>(EN_IMPL_HW_TBE) || implType == static_cast<int64_t>(EN_IMPL_CUSTOM_TBE) ||
        implType == static_cast<int64_t>(EN_IMPL_NON_PERSISTENT_CUSTOM_TBE)) {
      CHECK(axisValue.size() <= static_cast<size_t>(AXIS_C1), LOG_INFO("AxisValue is not correct!"), return true);
      int64_t hwc1 = static_cast<size_t>(axisValue[AXIS_C1] * axisValue[AXIS_H] * axisValue[AXIS_W]);
      newDimVec.push_back(hwc1);
      newDimVec.push_back(DivisionCeiling(static_cast<size_t>(axisValue[AXIS_N]), NI));
      newDimVec.push_back(NI);
      newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_C0]));
      newShape = ge::GeShape(newDimVec);
    } else {
      CHECK(axisValue.size() <= static_cast<size_t>(AXIS_W), LOG_INFO("AxisValue is not correct!"), return true);
      newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_N]));
      newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_C]));
      newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_H]));
      newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_W]));
      newShape = ge::GeShape(newDimVec);
    }
  }

  return true;
}

bool ShapeTransferAccordingToFormat::GetHWCNShapeByAxisValue(ge::GeShape &newShape, const int64_t &implType,
                                                             const vector<int64_t> &axisValue,
                                                             const vector<int64_t> &ndValue) {
  CHECK(axisValue.size() <= static_cast<size_t>(AXIS_W), LOG_INFO("AxisValue is not correct!"), return true);
  /* axisValue is initialized as a size 6 vector. */
  std::vector<int64_t> newDimVec;
  newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_H]));
  newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_W]));
  newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_C]));
  newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_N]));
  newShape = ge::GeShape(newDimVec);
  return true;
}

bool ShapeTransferAccordingToFormat::GetC1HWNCoC0ShapeByAxisValue(ge::GeShape &newShape, const int64_t &implType,
                                                                  const vector<int64_t> &axisValue,
                                                                  const vector<int64_t> &ndValue) {
  CHECK(axisValue.size() <= static_cast<size_t>(AXIS_Co), LOG_INFO("AxisValue is not correct!"), return true);
  /* axisValue is initialized as a size 6 vector. */
  std::vector<int64_t> newDimVec;
  newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_C1]));
  newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_H]));
  newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_W]));
  newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_N]));
  newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_Co]));
  newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_C0]));
  newShape = ge::GeShape(newDimVec);
  return true;
}

bool ShapeTransferAccordingToFormat::GetNzShapeByAxisValue(ge::GeShape &newShape, const int64_t &implType,
                                                           const vector<int64_t> &axisValue,
                                                           const vector<int64_t> &ndValue) {
  CHECK(ndValue.empty(), LOG_INFO("ndValue is empty!"), return true);
  CHECK(axisValue.empty() || axisValue.size() <= static_cast<size_t>(AXIS_C0),
        LOG_INFO("AxisValue is empty or its size %lu <= static_cast<size_t>(AXIS_C0[%d])", axisValue.size(), AXIS_C0),
        return true);
  uint32_t sizeOfOriginalVec = static_cast<uint32_t>(ndValue.size());
  if (sizeOfOriginalVec < MINIMUM_NZ_SHAPE_DIM_NUM) {
    LOG_INFO("ndValue's dim num is less than 2!");
    return true;
  }
  /* axisValue is initialized as a size 6 vector. */
  std::vector<int64_t> newDimVec = ndValue;

  /* sizeOfOriginalVec - 1 mean the last value of original vec
   * sizeOfOriginalVec - 2 mean the second last value of original vec */
  newDimVec[sizeOfOriginalVec - MINUS_VALUE_ONE] =
    DivisionCeiling(ndValue[sizeOfOriginalVec - MINUS_VALUE_TWO], (int64_t)SHAPE_NUMBER_16);

  newDimVec[sizeOfOriginalVec - MINUS_VALUE_TWO] =
    DivisionCeiling(ndValue[sizeOfOriginalVec - MINUS_VALUE_ONE], static_cast<size_t>(axisValue[AXIS_C0]));
  newDimVec.push_back(SHAPE_NUMBER_16);
  newDimVec.push_back(static_cast<size_t>(axisValue[AXIS_C0]));
  newShape = ge::GeShape(newDimVec);
  return true;
}

bool ShapeTransferAccordingToFormat::GetShapeAccordingToFormat(ShapeAndFormat &shapeAndFormatInfo, int64_t *c) {
  /* The default new shape is old shape */
  shapeAndFormatInfo.newShape = shapeAndFormatInfo.oldShape;
  if (shapeAndFormatInfo.oldFormat >= ge::FORMAT_RESERVED || shapeAndFormatInfo.newFormat >= ge::FORMAT_RESERVED) {
    LOG_ERROR("Old format %u or new format %u is invalid!", shapeAndFormatInfo.oldFormat, shapeAndFormatInfo.newFormat);
    return false;
  }

  if (shapeAndFormatInfo.currentDataType >= ge::DT_UNDEFINED) {
    LOG_ERROR("currentDataType %u is invalid!", shapeAndFormatInfo.currentDataType);
    return false;
  }
  AxisUtil *axisutil_object = new (std::nothrow) AxisUtil();
  if (axisutil_object == nullptr) {
    return false;
  }
  if (!axisutil_object->HasAxisValueFunc(shapeAndFormatInfo.oldFormat)) {
    delete axisutil_object;
    return true;
  }

  auto iterGetNewShapeFunc = getNewShapeFuncMap.find(shapeAndFormatInfo.newFormat);
  if (iterGetNewShapeFunc == getNewShapeFuncMap.end()) {
    LOG_INFO("Can not get new shape of new format %u!", shapeAndFormatInfo.newFormat);
    delete axisutil_object;
    return true;
  }
  LOG_INFO("Original format %u, new format %u", shapeAndFormatInfo.oldFormat, shapeAndFormatInfo.newFormat);
  GetNewShapeByAxisValueAndFormatPtr getNewShapeFunc = iterGetNewShapeFunc->second;
  if (getNewShapeFunc == nullptr) {
    LOG_ERROR("getNewShapeFunc is nullptr return false.");
    delete axisutil_object;
    return false;
  }
  std::vector<int64_t> axisValue;
  for (uint32_t i = 0; i < static_cast<size_t>(AXIS_BOTTOM); i++) {
    axisValue.push_back(1);
  }
  std::vector<int64_t> ndValue;
  uint32_t c0;
  if (mapOfDtypeAndC0.empty()) {
    c0 = SHAPE_NUMBER_16;
  } else {
    auto iterGetC0 = mapOfDtypeAndC0.find(shapeAndFormatInfo.currentDataType);
    if (iterGetC0 == mapOfDtypeAndC0.end()) {
      LOG_ERROR("Dtype is not support.");
      delete axisutil_object;
      return true;
    }
    c0 = iterGetC0->second;
  }

  // The value of C0 should be 4 while format is 5HD-4 or FRAZ-4
  if (shapeAndFormatInfo.newFormat == ge::FORMAT_NC1HWC0_C04) {
    c0 = SHAPE_DIM_VALUE_C04;
  }

  bool status = axisutil_object->GetAxisValueByOriginFormat(
    shapeAndFormatInfo.oldFormat, shapeAndFormatInfo.oldShape.GetDims(), c0, axisValue, ndValue);
  if (status != true && shapeAndFormatInfo.newFormat != ge::FORMAT_FRACTAL_NZ) {
    delete axisutil_object;
    return true;
  }
  delete axisutil_object;

  (void)(*getNewShapeFunc)(shapeAndFormatInfo.newShape, shapeAndFormatInfo.opImplType, axisValue, ndValue);
  if (c != nullptr) {
    *c = static_cast<size_t>(axisValue[AXIS_C]);
  }
  return true;
}
};  // namespace ge
