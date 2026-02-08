/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <vector>
#include "common/common_test.h"
#include "include/backend/common/ms_device_shape_transfer.h"
#include "include/utils/utils.h"

using namespace std;
namespace mindspore {
namespace trans {
class FormatTransTest : public UT::Common {
 public:
  FormatTransTest() = default;
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(FormatTransTest, nchw_to_hwcn) {
  uint16_t data[2 * 2 * 2 * 2] = {12581, 14220, 14937, 14302, 15004, 14951, 14694, 14564,
                                  14069, 14554, 10507, 14787, 13016, 15263, 14872, 10838};
  uint16_t res[2 * 2 * 2 * 2] = {12581, 14069, 15004, 13016, 14220, 14554, 14951, 15263,
                                 14937, 10507, 14694, 14872, 14302, 14787, 14564, 10838};
  size_t device_size = 32;
  auto trans_tmp = std::vector<uint8_t>(device_size);
  FormatArgs format_args{data,         device_size,  kOpFormat_NCHW,    kOpFormat_HWCN,
                         {2, 2, 2, 2}, {2, 2, 2, 2}, kNumberTypeFloat16};
  EXPECT_EQ(trans::TransFormat(format_args, trans_tmp.data(), nullptr, 0), true);
  for (size_t i = 0; i < sizeof(res) / sizeof(res[0]); i++) {
    EXPECT_EQ((reinterpret_cast<uint16_t *>(trans_tmp.data()))[i], res[i]);
  }
}

TEST_F(FormatTransTest, hwcn_to_nchw) {
  uint16_t data[2 * 2 * 2 * 2] = {12581, 14069, 15004, 13016, 14220, 14554, 14951, 15263,
                                  14937, 10507, 14694, 14872, 14302, 14787, 14564, 10838};

  uint16_t res[2 * 2 * 2 * 2] = {12581, 14220, 14937, 14302, 15004, 14951, 14694, 14564,
                                 14069, 14554, 10507, 14787, 13016, 15263, 14872, 10838};

  size_t device_size = 32;
  auto trans_tmp = std::vector<uint8_t>(device_size);
  FormatArgs format_args{data,         device_size,  kOpFormat_NCHW,    kOpFormat_HWCN,
                         {2, 2, 2, 2}, {2, 2, 2, 2}, kNumberTypeFloat16};
  EXPECT_EQ(trans::TransFormatFromDeviceToHost(format_args, trans_tmp.data()), true);
  for (size_t i = 0; i < sizeof(res) / sizeof(res[0]); i++) {
    EXPECT_EQ((reinterpret_cast<uint16_t *>(trans_tmp.data()))[i], res[i]);
  }
}

TEST_F(FormatTransTest, nchw_to_nhwc) {
  uint16_t data[2 * 2 * 2 * 2] = {11750, 13778, 15007, 15321, 15163, 13446, 15063, 14467,
                                  15056, 13284, 15219, 14797, 12684, 14288, 14855, 14799};
  uint16_t res[2 * 2 * 2 * 2] = {11750, 15163, 13778, 13446, 15007, 15063, 15321, 14467,
                                 15056, 12684, 13284, 14288, 15219, 14855, 14797, 14799};
  size_t device_size = 32;
  auto trans_tmp = std::vector<uint8_t>(device_size);
  FormatArgs format_args{data,         device_size,  kOpFormat_NCHW,    kOpFormat_NHWC,
                         {2, 2, 2, 2}, {2, 2, 2, 2}, kNumberTypeFloat16};
  EXPECT_EQ(trans::TransFormat(format_args, trans_tmp.data(), nullptr, 0), true);
  for (size_t i = 0; i < sizeof(res) / sizeof(res[0]); i++) {
    EXPECT_EQ((reinterpret_cast<uint16_t *>(trans_tmp.data()))[i], res[i]);
  }
}

TEST_F(FormatTransTest, nhwc_to_nchw) {
  uint16_t data[2 * 2 * 2 * 2] = {11750, 15163, 13778, 13446, 15007, 15063, 15321, 14467,
                                  15056, 12684, 13284, 14288, 15219, 14855, 14797, 14799};
  uint16_t res[2 * 2 * 2 * 2] = {11750, 13778, 15007, 15321, 15163, 13446, 15063, 14467,
                                 15056, 13284, 15219, 14797, 12684, 14288, 14855, 14799};

  size_t device_size = 32;
  auto trans_tmp = std::vector<uint8_t>(device_size);
  FormatArgs format_args{data,         device_size,  kOpFormat_NCHW,    kOpFormat_NHWC,
                         {2, 2, 2, 2}, {2, 2, 2, 2}, kNumberTypeFloat16};
  EXPECT_EQ(trans::TransFormatFromDeviceToHost(format_args, trans_tmp.data()), true);
  for (size_t i = 0; i < sizeof(res) / sizeof(res[0]); i++) {
    EXPECT_EQ((reinterpret_cast<uint16_t *>(trans_tmp.data()))[i], res[i]);
  }
}

class ShapeTransTest : public UT::Common {
 public:
  ShapeTransTest() = default;
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(ShapeTransTest, fraczn_rnn_device_shape) {
  std::vector<size_t> host_shape = {43, 120};
  std::string format = kOpFormat_FRACTAL_ZN_RNN;
  std::vector<int64_t> input_hidden_size = {13, 30};
  auto trans_shape = trans::TransShapeToDevice(host_shape, format, kNumberTypeFloat16, 1, input_hidden_size);
  const std::vector<size_t> expect_shape = {3, 8, 16, 16};
  EXPECT_EQ(trans_shape.size(), expect_shape.size());
  for (size_t i = 0; i < expect_shape.size(); i++) {
    EXPECT_EQ(trans_shape[i], expect_shape[i]);
  }
}

TEST_F(ShapeTransTest, nd_rnn_bias_device_shape) {
  std::vector<size_t> host_shape = {120};
  std::string format = kOpFormat_ND_RNN_BIAS;
  std::vector<int64_t> input_hidden_size = {13, 30};
  auto trans_shape = trans::TransShapeToDevice(host_shape, format, kNumberTypeFloat16, 1, input_hidden_size);
  std::vector<size_t> expect_shape = {128};
  EXPECT_EQ(trans_shape.size(), expect_shape.size());
  for (size_t i = 0; i < expect_shape.size(); i++) {
    EXPECT_EQ(trans_shape[i], expect_shape[i]);
  }
}

TEST_F(ShapeTransTest, fraczn_rnn_dynamic_device_shape) {
  std::vector<int64_t> host_shape = {-1, -1};
  std::string format = kOpFormat_FRACTAL_ZN_RNN;
  std::vector<int64_t> input_hidden_size = {13, 30};
  auto trans_shape = trans::TransShapeToDevice(host_shape, format, kNumberTypeFloat16, 1, input_hidden_size);
  const std::vector<int64_t> expect_shape = {-1, -1, 16, 16};
  EXPECT_EQ(trans_shape.size(), expect_shape.size());
  for (size_t i = 0; i < expect_shape.size(); i++) {
    EXPECT_EQ(trans_shape[i], expect_shape[i]);
  }
}

TEST_F(ShapeTransTest, nd_rnn_bias_dynamic_device_shape) {
  std::vector<int64_t> host_shape = {-1};
  std::string format = kOpFormat_ND_RNN_BIAS;
  std::vector<int64_t> input_hidden_size = {13, 30};
  auto trans_shape = trans::TransShapeToDevice(host_shape, format, kNumberTypeFloat16, 1, input_hidden_size);
  std::vector<int64_t> expect_shape = {-1};
  EXPECT_EQ(trans_shape.size(), expect_shape.size());
  for (size_t i = 0; i < expect_shape.size(); i++) {
    EXPECT_EQ(trans_shape[i], expect_shape[i]);
  }
}
}  // namespace trans
}  // namespace mindspore
