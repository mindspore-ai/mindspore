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
#include <memory>
#include <string>
#include "common/common.h"
#include "common/cvop_common.h"
#include "dataset/kernels/data/type_cast_op.h"
#include "dataset/core/client.h"
#include "dataset/core/cv_tensor.h"
#include "dataset/core/data_type.h"
#include "dataset/core/tensor.h"
#include "dataset/core/pybind_support.h"
#include "gtest/gtest.h"
#include "securec.h"
#include "dataset/util/de_error.h"

#define  MAX_INT_PRECISION 16777216  // float int precision is 16777216
using namespace mindspore::dataset;

namespace py = pybind11;


class MindDataTestTypeCast : public UT::Common {
 public:
    MindDataTestTypeCast() {}

    void SetUp() {
      GlobalInit();
    }
};

template<typename FROM, typename TO>
void testCast(std::vector<FROM> values, const DataType &from, const DataType &to) {
  std::shared_ptr<Tensor> t = std::make_shared<Tensor>(TensorShape({static_cast<int64_t>(values.size())}),
                                                       DataType(from),
                                                       reinterpret_cast<unsigned char *>(&values[0]));

  std::unique_ptr<TypeCastOp> op(new TypeCastOp(to));
  EXPECT_TRUE(op->OneToOne());
  std::shared_ptr<Tensor> output;
  EXPECT_TRUE(op->Compute(t, &output));
  ASSERT_TRUE(t->shape() == output->shape());
  ASSERT_TRUE(DataType(to)==output->type());
  MS_LOG(DEBUG) << *output << std::endl;
  auto out = output->begin<TO>();
  auto v = values.begin();
  for (; out != output->end<TO>(); out++, v++) {
    ASSERT_TRUE((*out) == static_cast<TO>(*v));
  }
}

TEST_F(MindDataTestTypeCast, CastFromUINT8) {
  std::vector<uint8_t> input{0, 10, 255};
  DataType input_format = DataType(DataType("uint8"));
  testCast<uint8_t, uint8_t>(input, input_format, DataType("uint8"));
  testCast<uint8_t, uint16_t>(input, input_format, DataType("uint16"));
  testCast<uint8_t, uint32_t>(input, input_format, DataType("uint32"));
  testCast<uint8_t, uint64_t>(input, input_format, DataType("uint64"));
  testCast<uint8_t, int8_t>(input, input_format, DataType("int8"));
  testCast<uint8_t, int16_t>(input, input_format, DataType("int16"));
  testCast<uint8_t, int32_t>(input, input_format, DataType("int32"));
  testCast<uint8_t, int64_t>(input, input_format, DataType("int64"));
  testCast<uint8_t, float16>(input, input_format, DataType("float16"));
  testCast<uint8_t, float>(input, input_format, DataType("float32"));
  testCast<uint8_t, double>(input, input_format, DataType("float64"));
  testCast<uint8_t, bool>(input, input_format, DataType("bool"));
}

TEST_F(MindDataTestTypeCast, CastFromINT64) {
  std::vector<int64_t> input{-9223372036854775806, 0, 9223372036854775807};
  DataType input_format = DataType("int64");
  testCast<int64_t, uint8_t>(input, input_format, DataType("uint8"));
  testCast<int64_t, uint16_t>(input, input_format, DataType("uint16"));
  testCast<int64_t, uint32_t>(input, input_format, DataType("uint32"));
  testCast<int64_t, uint64_t>(input, input_format, DataType("uint64"));
  testCast<int64_t, int8_t>(input, input_format, DataType("int8"));
  testCast<int64_t, int16_t>(input, input_format, DataType("int16"));
  testCast<int64_t, int32_t>(input, input_format, DataType("int32"));
  testCast<int64_t, int64_t>(input, input_format, DataType("int64"));
  testCast<int64_t, float16>(input, input_format, DataType("float16"));
  testCast<int64_t, float>(input, input_format, DataType("float32"));
  testCast<int64_t, double>(input, input_format, DataType("float64"));
  testCast<int64_t, bool>(input, input_format, DataType("bool"));
}

TEST_F(MindDataTestTypeCast, CastFromFLOAT64) {
  std::vector<double> input{(-1) * MAX_INT_PRECISION, 0, MAX_INT_PRECISION};
  DataType input_format = DataType("float64");
  testCast<double, uint8_t>(input, input_format, DataType("uint8"));
  testCast<double, uint16_t>(input, input_format, DataType("uint16"));
  testCast<double, uint32_t>(input, input_format, DataType("uint32"));
  testCast<double, uint64_t>(input, input_format, DataType("uint64"));
  testCast<double, int8_t>(input, input_format, DataType("int8"));
  testCast<double, int16_t>(input, input_format, DataType("int16"));
  testCast<double, int32_t>(input, input_format, DataType("int32"));
  testCast<double, int64_t>(input, input_format, DataType("int64"));
  testCast<double, float16>(input, input_format, DataType("float16"));
  testCast<double, float>(input, input_format, DataType("float32"));
  testCast<double, double>(input, input_format, DataType("float64"));
  testCast<double, bool>(input, input_format, DataType("bool"));
}

TEST_F(MindDataTestTypeCast, CastFromFLOAT16) {
  float16 min(0.0005);
  float16 zero(0);
  float16 max(32768);
  std::vector<float16> input{min, zero, max};
  DataType input_format = DataType("float16");
  testCast<float16, uint8_t>(input, input_format, DataType("uint8"));
  testCast<float16, uint16_t>(input, input_format, DataType("uint16"));
  testCast<float16, uint32_t>(input, input_format, DataType("uint32"));
  testCast<float16, uint64_t>(input, input_format, DataType("uint64"));
  testCast<float16, int8_t>(input, input_format, DataType("int8"));
  testCast<float16, int16_t>(input, input_format, DataType("int16"));
  testCast<float16, int32_t>(input, input_format, DataType("int32"));
  testCast<float16, int64_t>(input, input_format, DataType("int64"));
  testCast<float16, float16>(input, input_format, DataType("float16"));
  testCast<float16, float>(input, input_format, DataType("float32"));
  testCast<float16, double>(input, input_format, DataType("float64"));
  testCast<float16, bool>(input, input_format, DataType("bool"));
}
