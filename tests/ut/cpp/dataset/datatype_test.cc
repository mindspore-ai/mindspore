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
#include <string>
#include "./securec.h"
#include "minddata/dataset/core/data_type.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>
#include "minddata/dataset/include/constants.h"

using namespace mindspore::dataset;

class MindDataTestDatatype : public UT::Common {
 public:
    MindDataTestDatatype() = default;
};


TEST_F(MindDataTestDatatype, TestSizes) {
  uint8_t x = DataType::kTypeInfo[DataType::DE_BOOL].sizeInBytes_;
  DataType d = DataType(DataType::DE_BOOL);
  ASSERT_EQ(x, 1);
  ASSERT_EQ(d.SizeInBytes(), x);
  x = DataType::kTypeInfo[DataType::DE_INT8].sizeInBytes_;
  d = DataType(DataType::DE_INT8);
  ASSERT_EQ(x, 1);
  ASSERT_EQ(d.SizeInBytes(), x);
  x = DataType::kTypeInfo[DataType::DE_UINT8].sizeInBytes_;
  d = DataType(DataType::DE_UINT8);
  ASSERT_EQ(x, 1);
  ASSERT_EQ(d.SizeInBytes(), x);
  x = DataType::kTypeInfo[DataType::DE_INT16].sizeInBytes_;
  d = DataType(DataType::DE_INT16);
  ASSERT_EQ(x, 2);
  ASSERT_EQ(d.SizeInBytes(), x);
  x = DataType::kTypeInfo[DataType::DE_UINT16].sizeInBytes_;
  d = DataType(DataType::DE_UINT16);
  ASSERT_EQ(x, 2);
  ASSERT_EQ(d.SizeInBytes(), x);
  x = DataType::kTypeInfo[DataType::DE_INT32].sizeInBytes_;
  d = DataType(DataType::DE_INT32);
  ASSERT_EQ(x, 4);
  ASSERT_EQ(d.SizeInBytes(), x);
  x = DataType::kTypeInfo[DataType::DE_UINT32].sizeInBytes_;
  d = DataType(DataType::DE_UINT32);
  ASSERT_EQ(x, 4);
  ASSERT_EQ(d.SizeInBytes(), x);
  x = DataType::kTypeInfo[DataType::DE_INT64].sizeInBytes_;
  d = DataType(DataType::DE_INT64);
  ASSERT_EQ(x, 8);
  ASSERT_EQ(d.SizeInBytes(), x);
  x = DataType::kTypeInfo[DataType::DE_UINT64].sizeInBytes_;
  d = DataType(DataType::DE_UINT64);
  ASSERT_EQ(x, 8);
  ASSERT_EQ(d.SizeInBytes(), x);
  x = DataType::kTypeInfo[DataType::DE_FLOAT32].sizeInBytes_;
  d = DataType(DataType::DE_FLOAT32);
  ASSERT_EQ(x, 4);
  ASSERT_EQ(d.SizeInBytes(), x);
  x = DataType::kTypeInfo[DataType::DE_FLOAT64].sizeInBytes_;
  d = DataType(DataType::DE_FLOAT64);
  ASSERT_EQ(x, 8);
  ASSERT_EQ(d.SizeInBytes(), x);
}

void FromDT(DataType d, uint8_t cv_type, std::string str) {
  if (d == DataType::DE_UNKNOWN || d == DataType::DE_UINT32 || d == DataType::DE_UINT64 || d == DataType::DE_INT64) {
    ASSERT_EQ(d.AsCVType(), kCVInvalidType);
  } else {
    ASSERT_EQ(d.AsCVType(), cv_type);
  }
  ASSERT_EQ(d.ToString(), str);
}

TEST_F(MindDataTestDatatype, TestConstructors) {
  // Default constructor
  DataType d;
  ASSERT_EQ(d, DataType::DE_UNKNOWN);
  DataType d3 = DataType();
  ASSERT_EQ(d3, DataType::DE_UNKNOWN);

  // DataType(Type d)
  DataType d4(DataType::DE_FLOAT32);
  ASSERT_EQ(d4, DataType::DE_FLOAT32);
  DataType d5 = DataType(DataType::DE_UINT32);
  ASSERT_EQ(d5, DataType::DE_UINT32);

  // != operator
  ASSERT_NE(d4, d5);

  // == operator
  d5 = DataType(DataType::DE_FLOAT32);
  ASSERT_EQ(d4, d5);
}

TEST_F(MindDataTestDatatype, TestFromTypes) {
  FromDT(DataType(DataType::DE_BOOL), CV_8U, "bool");
  FromDT(DataType(DataType::DE_UINT8), CV_8U, "uint8");
  FromDT(DataType(DataType::DE_INT8), CV_8S, "int8");
  FromDT(DataType(DataType::DE_UINT16), CV_16U, "uint16");
  FromDT(DataType(DataType::DE_INT16), CV_16S, "int16");
  FromDT(DataType(DataType::DE_UINT32), 0, "uint32");
  FromDT(DataType(DataType::DE_INT32), CV_32S, "int32");
  FromDT(DataType(DataType::DE_UINT64), 0, "uint64");
  FromDT(DataType(DataType::DE_INT64), 0, "int64");
  FromDT(DataType(DataType::DE_FLOAT32), CV_32F, "float32");
  FromDT(DataType(DataType::DE_FLOAT64), CV_64F, "float64");
  FromDT(DataType(DataType::DE_UNKNOWN), CV_8U, "unknown");
}

TEST_F(MindDataTestDatatype, TestCompatible) {
  ASSERT_TRUE(DataType(DataType::DE_BOOL).IsCompatible<bool>());
  ASSERT_TRUE(DataType(DataType::DE_UINT8).IsCompatible<uint8_t>());
  ASSERT_TRUE(DataType(DataType::DE_INT8).IsCompatible<int8_t>());
  ASSERT_TRUE(DataType(DataType::DE_UINT16).IsCompatible<uint16_t>());
  ASSERT_TRUE(DataType(DataType::DE_INT16).IsCompatible<int16_t>());
  ASSERT_TRUE(DataType(DataType::DE_UINT32).IsCompatible<uint32_t>());
  ASSERT_TRUE(DataType(DataType::DE_INT32).IsCompatible<int32_t>());
  ASSERT_TRUE(DataType(DataType::DE_UINT64).IsCompatible<uint64_t>());
  ASSERT_TRUE(DataType(DataType::DE_INT64).IsCompatible<int64_t>());
  ASSERT_TRUE(DataType(DataType::DE_FLOAT32).IsCompatible<float>());
  ASSERT_TRUE(DataType(DataType::DE_FLOAT64).IsCompatible<double>());

  ASSERT_FALSE(DataType(DataType::DE_UINT8).IsCompatible<bool>());
  ASSERT_FALSE(DataType(DataType::DE_BOOL).IsCompatible<uint8_t>());
  ASSERT_FALSE(DataType(DataType::DE_UINT8).IsCompatible<int8_t>());
  ASSERT_FALSE(DataType(DataType::DE_UINT32).IsCompatible<uint16_t>());
  ASSERT_FALSE(DataType(DataType::DE_INT64).IsCompatible<double>());

  ASSERT_TRUE(DataType(DataType::DE_BOOL).IsLooselyCompatible<bool>());
  ASSERT_FALSE(DataType(DataType::DE_INT16).IsLooselyCompatible<bool>());

  ASSERT_TRUE(DataType(DataType::DE_UINT8).IsLooselyCompatible<uint8_t>());
  ASSERT_FALSE(DataType(DataType::DE_UINT64).IsLooselyCompatible<uint8_t>());

  ASSERT_TRUE(DataType(DataType::DE_UINT64).IsLooselyCompatible<uint64_t>());
  ASSERT_TRUE(DataType(DataType::DE_UINT32).IsLooselyCompatible<uint64_t>());
  ASSERT_TRUE(DataType(DataType::DE_UINT16).IsLooselyCompatible<uint64_t>());
  ASSERT_TRUE(DataType(DataType::DE_UINT8).IsLooselyCompatible<uint64_t>());


}

TEST_F(MindDataTestDatatype, TestCVTypes) {
  ASSERT_EQ(DataType::DE_UINT8, DataType::FromCVType(CV_8U).value());
  ASSERT_EQ(DataType::DE_UINT8, DataType::FromCVType(CV_8UC1).value());
  ASSERT_EQ(DataType::DE_UINT8, DataType::FromCVType(CV_8UC2).value());
  ASSERT_EQ(DataType::DE_UINT8, DataType::FromCVType(CV_8UC3).value());
  ASSERT_EQ(DataType::DE_UINT8, DataType::FromCVType(CV_8UC4).value());
  ASSERT_EQ(DataType::DE_UINT8, DataType::FromCVType(CV_8UC(5)).value());
  ASSERT_EQ(DataType::DE_INT8, DataType::FromCVType(CV_8S).value());
  ASSERT_EQ(DataType::DE_INT8, DataType::FromCVType(CV_8SC1).value());
  ASSERT_EQ(DataType::DE_INT8, DataType::FromCVType(CV_8SC2).value());
  ASSERT_EQ(DataType::DE_INT8, DataType::FromCVType(CV_8SC3).value());
  ASSERT_EQ(DataType::DE_INT8, DataType::FromCVType(CV_8SC(5)).value());
  ASSERT_EQ(DataType::DE_UINT16, DataType::FromCVType(CV_16U).value());
  ASSERT_EQ(DataType::DE_UINT16, DataType::FromCVType(CV_16UC1).value());
  ASSERT_EQ(DataType::DE_UINT16, DataType::FromCVType(CV_16UC2).value());
  ASSERT_EQ(DataType::DE_UINT16, DataType::FromCVType(CV_16UC3).value());
  ASSERT_EQ(DataType::DE_UINT16, DataType::FromCVType(CV_16UC4).value());
  ASSERT_EQ(DataType::DE_UINT16, DataType::FromCVType(CV_16UC(5)).value());
  ASSERT_EQ(DataType::DE_INT16, DataType::FromCVType(CV_16S).value());
  ASSERT_EQ(DataType::DE_INT16, DataType::FromCVType(CV_16SC1).value());
  ASSERT_EQ(DataType::DE_INT16, DataType::FromCVType(CV_16SC2).value());
  ASSERT_EQ(DataType::DE_INT16, DataType::FromCVType(CV_16SC3).value());
  ASSERT_EQ(DataType::DE_INT16, DataType::FromCVType(CV_16SC(5)).value());
  ASSERT_EQ(DataType::DE_INT32, DataType::FromCVType(CV_32S).value());
  ASSERT_EQ(DataType::DE_INT32, DataType::FromCVType(CV_32SC1).value());
  ASSERT_EQ(DataType::DE_INT32, DataType::FromCVType(CV_32SC2).value());
  ASSERT_EQ(DataType::DE_INT32, DataType::FromCVType(CV_32SC3).value());
  ASSERT_EQ(DataType::DE_INT32, DataType::FromCVType(CV_32SC(5)).value());
  ASSERT_EQ(DataType::DE_FLOAT32, DataType::FromCVType(CV_32F).value());
  ASSERT_EQ(DataType::DE_FLOAT32, DataType::FromCVType(CV_32FC1).value());
  ASSERT_EQ(DataType::DE_FLOAT32, DataType::FromCVType(CV_32FC2).value());
  ASSERT_EQ(DataType::DE_FLOAT32, DataType::FromCVType(CV_32FC3).value());
  ASSERT_EQ(DataType::DE_FLOAT32, DataType::FromCVType(CV_32FC(5)).value());
  ASSERT_EQ(DataType::DE_FLOAT64, DataType::FromCVType(CV_64F).value());
  ASSERT_EQ(DataType::DE_FLOAT64, DataType::FromCVType(CV_64FC1).value());
  ASSERT_EQ(DataType::DE_FLOAT64, DataType::FromCVType(CV_64FC2).value());
  ASSERT_EQ(DataType::DE_FLOAT64, DataType::FromCVType(CV_64FC3).value());
  ASSERT_EQ(DataType::DE_FLOAT64, DataType::FromCVType(CV_64FC(5)).value());
}
