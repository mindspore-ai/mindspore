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
#include <iostream>
#include <memory>
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "src/common/file_utils.h"
#include "mindspore/lite/nnacl/fp32/arithmetic.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/src/lite_kernel.h"
#include "include/errorcode.h"

namespace mindspore {

class TestArithmeticTestFp32 : public mindspore::CommonTest {
 public:
  TestArithmeticTestFp32() {}
  void PrepareInt(const std::vector<int> &input0_shape, const std::vector<int> &input1_shape, bool broadcast,
                  const std::vector<int> &output_shape, int *input0_data, int *input1_data, int *output_data, int type,
                  int act_type, const int thread_num);
  void TearDown() override;

 public:
  float err_tol = 1e-5;
  lite::Tensor in_tensor_0_;
  lite::Tensor in_tensor_1_;
  lite::Tensor out_tensor_;
  std::vector<lite::Tensor *> inputs_{&in_tensor_0_, &in_tensor_1_};
  std::vector<lite::Tensor *> outputs_{&out_tensor_};
  ArithmeticParameter param_;
  kernel::KernelKey desc_ = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt, schema::PrimitiveType_Eltwise};
  lite::InnerContext ctx_ = lite::InnerContext();
  kernel::KernelCreator creator_ = nullptr;
  kernel::LiteKernel *kernel_ = nullptr;
};

void TestArithmeticTestFp32::PrepareInt(const std::vector<int> &input0_shape, const std::vector<int> &input1_shape,
                                        bool broadcast, const std::vector<int> &output_shape, int *input0_data,
                                        int *input1_data, int *output_data, int type, int act_type,
                                        const int thread_num) {
  param_.broadcasting_ = true;
  param_.op_parameter_.type_ = type;
  param_.ndim_ = input0_shape.size();
  param_.activation_type_ = act_type;
  param_.broadcasting_ = broadcast;
  for (size_t i = 0; i < input0_shape.size(); ++i) {
    param_.in_shape0_[i] = input0_shape[i];
  }
  for (size_t i = 0; i < input1_shape.size(); ++i) {
    param_.in_shape1_[i] = input1_shape[i];
  }
  for (size_t i = 0; i < output_shape.size(); ++i) {
    param_.out_shape_[i] = output_shape[i];
  }

  in_tensor_0_.set_data_type(kNumberTypeInt);
  in_tensor_0_.set_data(input0_data);
  in_tensor_0_.set_shape(input0_shape);
  in_tensor_1_.set_data(input1_data);
  in_tensor_1_.set_shape(input1_shape);
  out_tensor_.set_data(output_data);
  out_tensor_.set_shape(output_shape);

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc_);
  ASSERT_NE(creator, nullptr);
  ctx_.thread_num_ = thread_num;
  ASSERT_EQ(lite::RET_OK, ctx_.Init());
  kernel_ = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(&param_), &ctx_, desc_, nullptr);
  ASSERT_NE(kernel_, nullptr);
}

void TestArithmeticTestFp32::TearDown() {
  in_tensor_0_.set_data(nullptr);
  in_tensor_1_.set_data(nullptr);
  out_tensor_.set_data(nullptr);
}

TEST_F(TestArithmeticTestFp32, AddTest) {
  auto add_param = new ArithmeticParameter();
  add_param->ndim_ = 4;
  add_param->in_shape0_[0] = 1;
  add_param->in_shape0_[1] = 2;
  add_param->in_shape0_[2] = 3;
  add_param->in_shape0_[3] = 4;
  add_param->in_shape1_[0] = 1;
  add_param->in_shape1_[1] = 1;
  add_param->in_shape1_[2] = 1;
  add_param->in_shape1_[3] = 4;
  add_param->out_shape_[0] = 1;
  add_param->out_shape_[1] = 2;
  add_param->out_shape_[2] = 3;
  add_param->out_shape_[3] = 4;

  /* 1x2x3x4 NHWC */
  std::vector<float> in = {12.216284, 3.3466918,  15.327419, 5.234958,  0.804376,   9.952188,  14.727955,  -8.080715,
                           13.71383,  8.055829,   6.5845337, -9.25232,  -4.24519,   11.550042, 9.262012,   1.2780352,
                           6.7263746, -3.9301445, 3.764492,  -8.602078, -3.3558068, 13.619035, -2.6694393, 3.2008505};
  auto in_ptr = in.data();
  std::vector<float> add = {0.9035316, 0.022212252, 0.3038014, 0.3478275};
  auto add_ptr = add.data();
  std::vector<float> correct_out = {13.119816,  3.368904,   15.631221,  5.5827856, 1.7079077, 9.9744,
                                    15.031756,  -7.7328877, 14.617362,  8.078041,  6.888335,  -8.904492,
                                    -3.3416586, 11.572254,  9.565813,   1.6258626, 7.629906,  -3.9079323,
                                    4.0682936,  -8.254251,  -2.4522753, 13.641247, -2.365638, 3.548678};
  auto correct_out_ptr = correct_out.data();

  int size = 1 * 2 * 3 * 4;
  auto out = new float[size];

  auto tile_data0 = new float[size];
  auto tile_data1 = new float[size];
  BroadcastAdd(in_ptr, add_ptr, tile_data0, tile_data1, out, size, add_param);
  ASSERT_EQ(0, CompareOutputData(out, correct_out_ptr, size, 0.00001));

  delete[] out;
  delete[] tile_data0;
  delete[] tile_data1;
  delete add_param;
}

TEST_F(TestArithmeticTestFp32, MulTest) {
  auto mul_param = new ArithmeticParameter();
  mul_param->ndim_ = 4;
  mul_param->in_shape0_[0] = 1;
  mul_param->in_shape0_[1] = 2;
  mul_param->in_shape0_[2] = 3;
  mul_param->in_shape0_[3] = 4;
  mul_param->in_shape1_[0] = 1;
  mul_param->in_shape1_[1] = 1;
  mul_param->in_shape1_[2] = 1;
  mul_param->in_shape1_[3] = 4;
  mul_param->out_shape_[0] = 1;
  mul_param->out_shape_[1] = 2;
  mul_param->out_shape_[2] = 3;
  mul_param->out_shape_[3] = 4;

  /* 1x2x3x4 NHWC */
  std::vector<float> in = {12.216284, 3.3466918,  15.327419, 5.234958,  0.804376,   9.952188,  14.727955,  -8.080715,
                           13.71383,  8.055829,   6.5845337, -9.25232,  -4.24519,   11.550042, 9.262012,   1.2780352,
                           6.7263746, -3.9301445, 3.764492,  -8.602078, -3.3558068, 13.619035, -2.6694393, 3.2008505};
  auto in_ptr = in.data();
  std::vector<float> add = {0.16771512, 0.7336843, 0.6768286, 0.4453379};
  auto add_ptr = add.data();
  std::vector<float> correct_out = {2.0488555,   2.4554152,  10.374036,   2.3313253, 0.13490601, 7.3017635,
                                    9.968302,    -3.5986485, 2.3000166,   5.910435,  4.4566007,  -4.120409,
                                    -0.71198255, 8.474085,   6.2687945,   0.5691575, 1.1281147,  -2.8834853,
                                    2.547916,    -3.8308315, -0.56281954, 9.992072,  -1.8067529, 1.42546};
  auto correct_out_ptr = correct_out.data();

  int size = 1 * 2 * 3 * 4;
  auto out = new float[size];

  auto tile_data0 = new float[size];
  auto tile_data1 = new float[size];
  BroadcastMul(in_ptr, add_ptr, tile_data0, tile_data1, out, size, mul_param);
  ASSERT_EQ(0, CompareOutputData(out, correct_out_ptr, size, 0.00001));

  delete[] out;
  delete[] tile_data0;
  delete[] tile_data1;
  delete mul_param;
}

TEST_F(TestArithmeticTestFp32, DivTest) {
  auto div_param = new ArithmeticParameter();
  div_param->ndim_ = 4;
  div_param->in_shape0_[0] = 1;
  div_param->in_shape0_[1] = 2;
  div_param->in_shape0_[2] = 3;
  div_param->in_shape0_[3] = 4;
  div_param->in_shape1_[0] = 1;
  div_param->in_shape1_[1] = 1;
  div_param->in_shape1_[2] = 1;
  div_param->in_shape1_[3] = 4;
  div_param->out_shape_[0] = 1;
  div_param->out_shape_[1] = 2;
  div_param->out_shape_[2] = 3;
  div_param->out_shape_[3] = 4;

  /* 1x2x3x4 NHWC */
  std::vector<float> in = {12.216284, 3.3466918,  15.327419, 5.234958,  0.804376,   9.952188,  14.727955,  -8.080715,
                           13.71383,  8.055829,   6.5845337, -9.25232,  -4.24519,   11.550042, 9.262012,   1.2780352,
                           6.7263746, -3.9301445, 3.764492,  -8.602078, -3.3558068, 13.619035, -2.6694393, 3.2008505};
  auto in_ptr = in.data();
  std::vector<float> add = {1.6771512, -7.336843, 0.6768286, 4.453379};
  auto add_ptr = add.data();
  std::vector<float> correct_out = {7.28394912,  -0.45614875, 22.64593872, 1.17550247,  0.47960852,  -1.35646735,
                                    21.76024329, -1.8145132,  8.17685967,  -1.09799665, 9.72850985,  -2.07759546,
                                    -2.53119099, -1.5742523,  13.68442764, 0.28698101,  4.01059523,  0.53567243,
                                    5.56195764,  -1.93158453, -2.000897,   -1.85625275, -3.94404034, 0.71874648};
  auto correct_out_ptr = correct_out.data();

  int size = 1 * 1 * 3 * 4;
  auto out = new float[size];

  auto tile_data0 = new float[size];
  auto tile_data1 = new float[size];
  BroadcastDiv(in_ptr, add_ptr, tile_data0, tile_data1, out, size, div_param);
  ASSERT_EQ(0, CompareOutputData(out, correct_out_ptr, size, 0.00001));

  delete[] out;
  delete[] tile_data0;
  delete[] tile_data1;
  delete div_param;
}

TEST_F(TestArithmeticTestFp32, DivTest2) {
  std::vector<float> in0 = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
  std::vector<float> in1 = {5, 10, 2, 8, 2, 3, 7, 80, 45, 20};
  std::vector<float> correct_out = {2, 2, 15, 5, 25, 20, 10, 1, 2, 5};
  constexpr int kOutSize = 10;
  float out[kOutSize];
  ElementDiv(in0.data(), in1.data(), out, kOutSize);
  std::cout << "out: ";
  for (int i = 0; i < kOutSize; ++i) {
    std::cout << out[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(out, correct_out.data(), kOutSize, 0.00001));
}

TEST_F(TestArithmeticTestFp32, FloorDivTest) {
  auto fdiv_param = new ArithmeticParameter();
  fdiv_param->ndim_ = 4;
  fdiv_param->in_shape0_[0] = 1;
  fdiv_param->in_shape0_[1] = 1;
  fdiv_param->in_shape0_[2] = 3;
  fdiv_param->in_shape0_[3] = 4;
  fdiv_param->in_shape1_[0] = 1;
  fdiv_param->in_shape1_[1] = 1;
  fdiv_param->in_shape1_[2] = 1;
  fdiv_param->in_shape1_[3] = 4;
  fdiv_param->out_shape_[0] = 1;
  fdiv_param->out_shape_[1] = 1;
  fdiv_param->out_shape_[2] = 3;
  fdiv_param->out_shape_[3] = 4;

  /* 1x2x3x4 NHWC */
  std::vector<float> in = {1.1, -1.1, 3.123, -5.432, 0.1234, -0.0312, 12.1, 21.1, 9.1, 9.0, -100, 0.1};
  auto in_ptr = in.data();
  std::vector<float> add = {1, 3, 2, 0.3};
  auto add_ptr = add.data();
  std::vector<float> correct_out = {1, -1, 1, -19, 0, -1, 6, 70, 9, 3, -50, 0};
  auto correct_out_ptr = correct_out.data();

  int size = 1 * 1 * 3 * 4;
  auto out = new float[size];

  auto tile_data0 = new float[size];
  auto tile_data1 = new float[size];
  int ret = BroadcastFloorDiv(in_ptr, add_ptr, tile_data0, tile_data1, out, size, fdiv_param);
  EXPECT_EQ(ret, 0);
  ASSERT_EQ(0, CompareOutputData(out, correct_out_ptr, size, 0.00001));

  delete[] out;
  delete[] tile_data0;
  delete[] tile_data1;
  delete fdiv_param;
}

TEST_F(TestArithmeticTestFp32, FloorModTest) {
  auto fmod_param = new ArithmeticParameter();
  fmod_param->ndim_ = 4;
  fmod_param->in_shape0_[0] = 1;
  fmod_param->in_shape0_[1] = 1;
  fmod_param->in_shape0_[2] = 3;
  fmod_param->in_shape0_[3] = 4;
  fmod_param->in_shape1_[0] = 1;
  fmod_param->in_shape1_[1] = 1;
  fmod_param->in_shape1_[2] = 1;
  fmod_param->in_shape1_[3] = 4;
  fmod_param->out_shape_[0] = 1;
  fmod_param->out_shape_[1] = 1;
  fmod_param->out_shape_[2] = 3;
  fmod_param->out_shape_[3] = 4;

  /* 1x2x3x4 NHWC */
  std::vector<float> in = {1.1, -1.1, 3.123, -5.432, 0.1234, -0.0312, 12.1, 21.1, 9.1, 9.0, -100, 0.1};
  auto in_ptr = in.data();
  std::vector<float> add = {1, 3, 2, 0.3};
  auto add_ptr = add.data();
  std::vector<float> correct_out = {0.100000, 1.900000, 1.123000, 0.268000, 0.123400, 2.968800,
                                    0.100000, 0.100000, 0.100000, 0.000000, 0.000000, 0.100000};
  auto correct_out_ptr = correct_out.data();

  int size = 1 * 1 * 3 * 4;
  auto out = new float[size];

  auto tile_data0 = new float[size];
  auto tile_data1 = new float[size];
  int ret = BroadcastFloorMod(in_ptr, add_ptr, tile_data0, tile_data1, out, size, fmod_param);
  EXPECT_EQ(ret, 0);
  ASSERT_EQ(0, CompareOutputData(out, correct_out_ptr, size, 0.00001));

  delete[] out;
  delete[] tile_data0;
  delete[] tile_data1;
  delete fmod_param;
}

TEST_F(TestArithmeticTestFp32, LogicalAndTest) {
  auto logical_and_param = new ArithmeticParameter();
  logical_and_param->ndim_ = 4;
  logical_and_param->in_shape0_[0] = 1;
  logical_and_param->in_shape0_[1] = 2;
  logical_and_param->in_shape0_[2] = 3;
  logical_and_param->in_shape0_[3] = 4;
  logical_and_param->in_shape1_[0] = 1;
  logical_and_param->in_shape1_[1] = 1;
  logical_and_param->in_shape1_[2] = 1;
  logical_and_param->in_shape1_[3] = 4;
  logical_and_param->out_shape_[0] = 1;
  logical_and_param->out_shape_[1] = 2;
  logical_and_param->out_shape_[2] = 3;
  logical_and_param->out_shape_[3] = 4;

  /* 1x2x3x4 NHWC */
  std::vector<float> in = {12.216284, 3.3466918,  15.327419, 5.234958,  0,          9.952188,  14.727955,  -8.080715,
                           13.71383,  8.055829,   6.5845337, -9.25232,  -4.24519,   11.550042, 9.262012,   1.2780352,
                           6.7263746, -3.9301445, 3.764492,  -8.602078, -3.3558068, 13.619035, -2.6694393, 3.2008505};
  auto in_ptr = in.data();
  std::vector<float> add = {1.6771512, -7.336843, 0, 4.453379};
  auto add_ptr = add.data();
  std::vector<float> correct_out = {1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1};
  auto correct_out_ptr = correct_out.data();
  int size = 1 * 2 * 3 * 4;

  auto out = new float[size];
  auto tile_data0 = new float[size];
  auto tile_data1 = new float[size];
  BroadcastLogicalAnd(in_ptr, add_ptr, tile_data0, tile_data1, out, size, logical_and_param);
  ASSERT_EQ(0, CompareOutputData(out, correct_out_ptr, size, 0.00001));

  delete[] out;
  delete[] tile_data0;
  delete[] tile_data1;
  delete logical_and_param;
}

TEST_F(TestArithmeticTestFp32, LogicalOrTest) {
  auto logical_or_param = new ArithmeticParameter();
  logical_or_param->ndim_ = 4;
  logical_or_param->in_shape0_[0] = 1;
  logical_or_param->in_shape0_[1] = 2;
  logical_or_param->in_shape0_[2] = 3;
  logical_or_param->in_shape0_[3] = 4;
  logical_or_param->in_shape1_[0] = 1;
  logical_or_param->in_shape1_[1] = 1;
  logical_or_param->in_shape1_[2] = 1;
  logical_or_param->in_shape1_[3] = 4;
  logical_or_param->out_shape_[0] = 1;
  logical_or_param->out_shape_[1] = 2;
  logical_or_param->out_shape_[2] = 3;
  logical_or_param->out_shape_[3] = 4;

  /* 1x2x3x4 NHWC */
  std::vector<float> in = {12.216284, 3.3466918,  15.327419, 5.234958, 0.804376,   0,         14.727955,  -8.080715,
                           13.71383,  8.055829,   6.5845337, -9.25232, -4.24519,   11.550042, 9.262012,   1.2780352,
                           6.7263746, -3.9301445, 3.764492,  0,        -3.3558068, 13.619035, -2.6694393, 3.2008505};

  auto in_ptr = in.data();
  std::vector<float> add = {1.6771512, 0, 0.6768286, 0};
  auto add_ptr = add.data();
  std::vector<float> correct_out = {1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1};
  auto correct_out_ptr = correct_out.data();

  int size = 1 * 2 * 3 * 4;

  auto out = new float[size];
  auto tile_data0 = new float[size];
  auto tile_data1 = new float[size];
  BroadcastLogicalOr(in_ptr, add_ptr, tile_data0, tile_data1, out, size, logical_or_param);
  ASSERT_EQ(0, CompareOutputData(out, correct_out_ptr, size, 0.00001));

  delete[] out;
  delete[] tile_data0;
  delete[] tile_data1;
  delete logical_or_param;
}

TEST_F(TestArithmeticTestFp32, MaximumTest) {
  auto maximum_param = new ArithmeticParameter();
  maximum_param->ndim_ = 4;
  maximum_param->in_shape0_[0] = 1;
  maximum_param->in_shape0_[1] = 2;
  maximum_param->in_shape0_[2] = 3;
  maximum_param->in_shape0_[3] = 4;
  maximum_param->in_shape1_[0] = 1;
  maximum_param->in_shape1_[1] = 1;
  maximum_param->in_shape1_[2] = 1;
  maximum_param->in_shape1_[3] = 4;
  maximum_param->out_shape_[0] = 1;
  maximum_param->out_shape_[1] = 2;
  maximum_param->out_shape_[2] = 3;
  maximum_param->out_shape_[3] = 4;

  /* 1x2x3x4 NHWC */
  std::vector<float> in = {12.216284, 3.3466918,  15.327419, 5.234958, 0.804376,   0,         14.727955,  -8.080715,
                           13.71383,  8.055829,   6.5845337, -9.25232, -4.24519,   11.550042, 9.262012,   1.2780352,
                           6.7263746, -3.9301445, 3.764492,  0,        -3.3558068, 13.619035, -2.6694393, 3.2008505};

  auto in_ptr = in.data();
  std::vector<float> add = {1.6771512, 6.34876, 3.6768286, 2.936284};
  auto add_ptr = add.data();
  std::vector<float> correct_out = {12.216284, 6.34876,   15.327419, 5.234958,  1.6771512, 6.34876,
                                    14.727955, 2.936284,  13.71383,  8.055829,  6.5845337, 2.936284,
                                    1.6771512, 11.550042, 9.262012,  2.936284,  6.7263746, 6.34876,
                                    3.764492,  2.93628,   1.6771512, 13.619035, 3.6768286, 3.2008505};
  auto correct_out_ptr = correct_out.data();

  int size = 1 * 2 * 3 * 4;

  auto out = new float[size];
  auto tile_data0 = new float[size];
  auto tile_data1 = new float[size];
  BroadcastMaximum(in_ptr, add_ptr, tile_data0, tile_data1, out, size, maximum_param);
  ASSERT_EQ(0, CompareOutputData(out, correct_out_ptr, size, 0.00001));

  delete[] out;
  delete[] tile_data0;
  delete[] tile_data1;
  delete maximum_param;
}

TEST_F(TestArithmeticTestFp32, MinimumTest) {
  auto minimum_param = new ArithmeticParameter();
  minimum_param->ndim_ = 4;
  minimum_param->in_shape0_[0] = 1;
  minimum_param->in_shape0_[1] = 2;
  minimum_param->in_shape0_[2] = 3;
  minimum_param->in_shape0_[3] = 4;
  minimum_param->in_shape1_[0] = 1;
  minimum_param->in_shape1_[1] = 1;
  minimum_param->in_shape1_[2] = 1;
  minimum_param->in_shape1_[3] = 4;
  minimum_param->out_shape_[0] = 1;
  minimum_param->out_shape_[1] = 2;
  minimum_param->out_shape_[2] = 3;
  minimum_param->out_shape_[3] = 4;

  /* 1x2x3x4 NHWC */
  std::vector<float> in = {12.216284, 3.3466918,  15.327419, 5.234958, 0.804376,   0,         14.727955,  -8.080715,
                           13.71383,  8.055829,   6.5845337, -9.25232, -4.24519,   11.550042, 9.262012,   1.2780352,
                           6.7263746, -3.9301445, 3.764492,  0,        -3.3558068, 13.619035, -2.6694393, 3.2008505};

  auto in_ptr = in.data();
  std::vector<float> add = {1.6771512, 6.34876, 3.6768286, 2.936284};
  auto add_ptr = add.data();
  std::vector<float> correct_out = {1.6771512, 3.3466918, 3.6768286,  2.936284,  0.804376,   0,
                                    3.6768286, -8.080715, 1.6771512,  6.34876,   3.6768286,  -9.25232,
                                    -4.24519,  6.34876,   3.6768286,  1.2780352, 1.6771512,  -3.9301445,
                                    3.6768286, 0,         -3.3558068, 6.34876,   -2.6694393, 2.936284};
  auto correct_out_ptr = correct_out.data();

  int size = 1 * 2 * 3 * 4;

  auto out = new float[size];
  auto tile_data0 = new float[size];
  auto tile_data1 = new float[size];
  BroadcastMinimum(in_ptr, add_ptr, tile_data0, tile_data1, out, size, minimum_param);
  ASSERT_EQ(0, CompareOutputData(out, correct_out_ptr, size, 0.00001));

  delete[] out;
  delete[] tile_data0;
  delete[] tile_data1;
  delete minimum_param;
}

TEST_F(TestArithmeticTestFp32, SquaredDifferenceTest) {
  auto add_param = new ArithmeticParameter();
  add_param->ndim_ = 3;
  add_param->in_shape0_[0] = 2;
  add_param->in_shape0_[1] = 3;
  add_param->in_shape0_[2] = 2;
  add_param->in_shape1_[0] = 2;
  add_param->in_shape1_[1] = 1;
  add_param->in_shape1_[2] = 2;
  add_param->out_shape_[0] = 2;
  add_param->out_shape_[1] = 3;
  add_param->out_shape_[2] = 2;

  /* 1x2x3x4 NHWC */
  std::vector<float> in = {10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25};
  auto in_ptr = in.data();
  std::vector<float> add = {30, 31, 32, 33};
  auto add_ptr = add.data();
  std::vector<float> correct_out = {400, 400, 324, 324, 256, 256, 144, 144, 100, 100, 64, 64};
  auto correct_out_ptr = correct_out.data();

  int size = 2 * 3 * 2;
  auto out = new float[size];

  auto tile_data0 = new float[size];
  auto tile_data1 = new float[size];
  BroadcastSub(in_ptr, add_ptr, tile_data0, tile_data1, out, size, add_param);
  ElementMul(out, out, out, size);
  ASSERT_EQ(0, CompareOutputData(out, correct_out_ptr, size, 0.00001));

  delete[] out;
  delete[] tile_data0;
  delete[] tile_data1;
  delete add_param;
}

TEST_F(TestArithmeticTestFp32, MulFp32) {
  std::vector<lite::Tensor *> inputs_tensor;
  std::vector<lite::Tensor *> outputs_tensor;

  ArithmeticParameter mul_param;
  mul_param.broadcasting_ = true;
  mul_param.op_parameter_.type_ = schema::PrimitiveType_Mul;
  mul_param.ndim_ = 4;
  mul_param.in_shape0_[0] = 1;
  mul_param.in_shape0_[1] = 2;
  mul_param.in_shape0_[2] = 3;
  mul_param.in_shape0_[3] = 4;
  mul_param.in_shape1_[0] = 1;
  mul_param.in_shape1_[1] = 1;
  mul_param.in_shape1_[2] = 1;
  mul_param.in_shape1_[3] = 4;
  mul_param.out_shape_[0] = 1;
  mul_param.out_shape_[1] = 2;
  mul_param.out_shape_[2] = 3;
  mul_param.out_shape_[3] = 4;

  /* 1x2x3x4 NHWC */
  std::vector<float> input0 = {12.216284, 3.3466918, 15.327419,  5.234958,  0.804376,   9.952188,
                               14.727955, -8.080715, 13.71383,   8.055829,  6.5845337,  -9.25232,
                               -4.24519,  11.550042, 9.262012,   1.2780352, 6.7263746,  -3.9301445,
                               3.764492,  -8.602078, -3.3558068, 13.619035, -2.6694393, 3.2008505};
  std::vector<int> input0_shape = {1, 2, 3, 4};
  std::vector<float> input1 = {0.16771512, 0.7336843, 0.6768286, 0.4453379};
  std::vector<int> input1_shape = {1, 1, 1, 4};

  lite::Tensor input0_tensor;
  lite::Tensor input1_tensor;
  input0_tensor.set_data_type(kNumberTypeFloat32);
  input0_tensor.set_data(input0.data());
  input1_tensor.set_data(input1.data());
  input0_tensor.set_shape(input0_shape);
  input1_tensor.set_shape(input1_shape);
  inputs_tensor.push_back(&input0_tensor);
  inputs_tensor.push_back(&input1_tensor);

  std::vector<float> output(24);
  std::vector<int> output_shape = {1, 2, 3, 4};

  lite::Tensor output0_tensor;
  outputs_tensor.push_back(&output0_tensor);
  output0_tensor.set_data(output.data());
  output0_tensor.set_shape(output_shape);

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Eltwise};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  lite::InnerContext ctx;
  ctx.thread_num_ = 3;
  ASSERT_EQ(lite::RET_OK, ctx.Init());
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&mul_param), &ctx, desc, nullptr);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor.shape();
  kernel->Run();

  std::vector<float> correct_out = {2.0488555,   2.4554152,  10.374036,   2.3313253, 0.13490601, 7.3017635,
                                    9.968302,    -3.5986485, 2.3000166,   5.910435,  4.4566007,  -4.120409,
                                    -0.71198255, 8.474085,   6.2687945,   0.5691575, 1.1281147,  -2.8834853,
                                    2.547916,    -3.8308315, -0.56281954, 9.992072,  -1.8067529, 1.42546};
  auto correct_out_ptr = correct_out.data();

  ASSERT_EQ(0, CompareOutputData(output.data(), correct_out_ptr, 24, 0.00001));

  input0_tensor.set_data(nullptr);
  input1_tensor.set_data(nullptr);
  output0_tensor.set_data(nullptr);
}

TEST_F(TestArithmeticTestFp32, MulReluFp32) {
  std::vector<lite::Tensor *> inputs_tensor;
  std::vector<lite::Tensor *> outputs_tensor;

  ArithmeticParameter mul_param;
  mul_param.broadcasting_ = true;
  mul_param.op_parameter_.type_ = schema::PrimitiveType_Mul;
  mul_param.ndim_ = 4;
  mul_param.activation_type_ = schema::ActivationType_RELU;
  mul_param.in_shape0_[0] = 1;
  mul_param.in_shape0_[1] = 2;
  mul_param.in_shape0_[2] = 3;
  mul_param.in_shape0_[3] = 4;
  mul_param.in_shape1_[0] = 1;
  mul_param.in_shape1_[1] = 1;
  mul_param.in_shape1_[2] = 1;
  mul_param.in_shape1_[3] = 4;
  mul_param.out_shape_[0] = 1;
  mul_param.out_shape_[1] = 2;
  mul_param.out_shape_[2] = 3;
  mul_param.out_shape_[3] = 4;

  /* 1x2x3x4 NHWC */
  std::vector<float> input0 = {12.216284, 3.3466918, 15.327419,  5.234958,  0.804376,   9.952188,
                               14.727955, -8.080715, 13.71383,   8.055829,  6.5845337,  -9.25232,
                               -4.24519,  11.550042, 9.262012,   1.2780352, 6.7263746,  -3.9301445,
                               3.764492,  -8.602078, -3.3558068, 13.619035, -2.6694393, 3.2008505};
  std::vector<int> input0_shape = {1, 2, 3, 4};
  std::vector<float> input1 = {0.16771512, 0.7336843, 0.6768286, 0.4453379};
  std::vector<int> input1_shape = {1, 1, 1, 4};

  lite::Tensor input0_tensor;
  lite::Tensor input1_tensor;
  input0_tensor.set_data_type(kNumberTypeFloat32);
  input0_tensor.set_data(input0.data());
  input1_tensor.set_data(input1.data());
  input0_tensor.set_shape(input0_shape);
  input1_tensor.set_shape(input1_shape);
  inputs_tensor.push_back(&input0_tensor);
  inputs_tensor.push_back(&input1_tensor);

  std::vector<float> output(24);
  std::vector<int> output_shape = {1, 2, 3, 4};

  lite::Tensor output0_tensor;
  outputs_tensor.push_back(&output0_tensor);
  output0_tensor.set_data(output.data());
  output0_tensor.set_shape(output_shape);

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Eltwise};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  lite::InnerContext ctx;
  ctx.thread_num_ = 3;
  ASSERT_EQ(lite::RET_OK, ctx.Init());
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&mul_param), &ctx, desc, nullptr);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor.shape();
  kernel->Run();

  std::vector<float> correct_out = {2.0488555, 2.4554152, 10.374036, 2.3313253, 0.13490601, 7.3017635,
                                    9.968302,  0,         2.3000166, 5.910435,  4.4566007,  0,
                                    0,         8.474085,  6.2687945, 0.5691575, 1.1281147,  0,
                                    2.547916,  0,         0,         9.992072,  0,          1.42546};
  auto correct_out_ptr = correct_out.data();

  ASSERT_EQ(0, CompareOutputData(output.data(), correct_out_ptr, 24, 0.00001));

  input0_tensor.set_data(nullptr);
  input1_tensor.set_data(nullptr);
  output0_tensor.set_data(nullptr);
}

TEST_F(TestArithmeticTestFp32, MulRelu6Fp32) {
  std::vector<lite::Tensor *> inputs_tensor;
  std::vector<lite::Tensor *> outputs_tensor;

  ArithmeticParameter mul_param;
  mul_param.broadcasting_ = true;
  mul_param.op_parameter_.type_ = schema::PrimitiveType_Mul;
  mul_param.ndim_ = 4;
  mul_param.activation_type_ = schema::ActivationType_RELU6;
  mul_param.in_shape0_[0] = 1;
  mul_param.in_shape0_[1] = 2;
  mul_param.in_shape0_[2] = 3;
  mul_param.in_shape0_[3] = 4;
  mul_param.in_shape1_[0] = 1;
  mul_param.in_shape1_[1] = 1;
  mul_param.in_shape1_[2] = 1;
  mul_param.in_shape1_[3] = 4;
  mul_param.out_shape_[0] = 1;
  mul_param.out_shape_[1] = 2;
  mul_param.out_shape_[2] = 3;
  mul_param.out_shape_[3] = 4;

  /* 1x2x3x4 NHWC */
  std::vector<float> input0 = {12.216284, 3.3466918, 15.327419,  5.234958,  0.804376,   9.952188,
                               14.727955, -8.080715, 13.71383,   8.055829,  6.5845337,  -9.25232,
                               -4.24519,  11.550042, 9.262012,   1.2780352, 6.7263746,  -3.9301445,
                               3.764492,  -8.602078, -3.3558068, 13.619035, -2.6694393, 3.2008505};
  std::vector<int> input0_shape = {1, 2, 3, 4};
  std::vector<float> input1 = {0.16771512, 0.7336843, 0.6768286, 0.4453379};
  std::vector<int> input1_shape = {1, 1, 1, 4};

  lite::Tensor input0_tensor;
  lite::Tensor input1_tensor;
  input0_tensor.set_data_type(kNumberTypeFloat32);
  input0_tensor.set_data(input0.data());
  input1_tensor.set_data(input1.data());
  input0_tensor.set_shape(input0_shape);
  input1_tensor.set_shape(input1_shape);
  inputs_tensor.push_back(&input0_tensor);
  inputs_tensor.push_back(&input1_tensor);

  std::vector<float> output(24);
  std::vector<int> output_shape = {1, 2, 3, 4};

  lite::Tensor output0_tensor;
  outputs_tensor.push_back(&output0_tensor);
  output0_tensor.set_data(output.data());
  output0_tensor.set_shape(output_shape);

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Eltwise};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  lite::InnerContext ctx;
  ctx.thread_num_ = 3;
  ASSERT_EQ(lite::RET_OK, ctx.Init());
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&mul_param), &ctx, desc, nullptr);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor.shape();
  kernel->Run();

  std::vector<float> correct_out = {2.0488555, 2.4554152, 6,         2.3313253, 0.13490601, 6, 6, 0,
                                    2.3000166, 5.910435,  4.4566007, 0,         0,          6, 6, 0.5691575,
                                    1.1281147, 0,         2.547916,  0,         0,          6, 0, 1.42546};
  auto correct_out_ptr = correct_out.data();

  ASSERT_EQ(0, CompareOutputData(output.data(), correct_out_ptr, 24, 0.00001));

  input0_tensor.set_data(nullptr);
  input1_tensor.set_data(nullptr);
  output0_tensor.set_data(nullptr);
}

TEST_F(TestArithmeticTestFp32, MulInt0) {
  std::vector<int> input0_shape{1, 2, 2, 3};
  std::vector<int> input1_shape{1, 1, 1, 3};
  bool broadcast = true;
  std::vector<int> output_shape{1, 2, 2, 3};
  int in0_data[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  int in1_data[3] = {3, 2, 1};
  int out_data[12] = {0};
  schema::PrimitiveType type = schema::PrimitiveType_Mul;
  int act_type = schema::ActivationType_NO_ACTIVATION;
  int thread_num = 2;
  desc_.type = type;
  PrepareInt(input0_shape, input1_shape, broadcast, output_shape, in0_data, in1_data, out_data, type, act_type,
             thread_num);
  kernel_->Run();

  int correct_data[12] = {0, 2, 2, 9, 8, 5, 18, 14, 8, 27, 20, 11};

  ASSERT_EQ(0, CompareOutputData(out_data, correct_data, 12, err_tol));
}

TEST_F(TestArithmeticTestFp32, MulInt1) {
  std::vector<int> input0_shape{1, 2, 2, 3};
  std::vector<int> input1_shape{1};
  bool broadcast = true;
  std::vector<int> output_shape{1, 2, 2, 3};
  int in0_data[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  int in1_data[1] = {2};
  int out_data[12] = {0};
  schema::PrimitiveType type = schema::PrimitiveType_Mul;
  int act_type = schema::ActivationType_NO_ACTIVATION;
  int thread_num = 2;
  desc_.type = type;
  PrepareInt(input0_shape, input1_shape, broadcast, output_shape, in0_data, in1_data, out_data, type, act_type,
             thread_num);
  kernel_->Run();

  int correct_data[12] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22};

  ASSERT_EQ(0, CompareOutputData(out_data, correct_data, 12, err_tol));
}

TEST_F(TestArithmeticTestFp32, MulInt2) {
  std::vector<int> input0_shape{1};
  std::vector<int> input1_shape{1, 2, 2, 3};
  bool broadcast = true;
  std::vector<int> output_shape{1, 2, 2, 3};
  int in0_data[1] = {2};
  int in1_data[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  int out_data[12] = {0};
  schema::PrimitiveType type = schema::PrimitiveType_Mul;
  int act_type = schema::ActivationType_NO_ACTIVATION;
  int thread_num = 2;
  desc_.type = type;
  PrepareInt(input0_shape, input1_shape, broadcast, output_shape, in0_data, in1_data, out_data, type, act_type,
             thread_num);
  kernel_->Run();

  int correct_data[12] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22};

  ASSERT_EQ(0, CompareOutputData(out_data, correct_data, 12, err_tol));
}

TEST_F(TestArithmeticTestFp32, MulInt3) {
  std::vector<int> input0_shape{1, 2, 2, 3};
  std::vector<int> input1_shape{1, 2, 2, 3};
  bool broadcast = false;
  std::vector<int> output_shape{1, 2, 2, 3};
  int in0_data[12] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  int in1_data[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  int out_data[12] = {0};
  schema::PrimitiveType type = schema::PrimitiveType_Mul;
  int act_type = schema::ActivationType_NO_ACTIVATION;
  int thread_num = 2;
  desc_.type = type;
  PrepareInt(input0_shape, input1_shape, broadcast, output_shape, in0_data, in1_data, out_data, type, act_type,
             thread_num);
  kernel_->Run();

  int correct_data[12] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22};

  ASSERT_EQ(0, CompareOutputData(out_data, correct_data, 12, err_tol));
}

TEST_F(TestArithmeticTestFp32, MulReluInt0) {
  std::vector<int> input0_shape{1, 2, 2, 3};
  std::vector<int> input1_shape{1, 1, 1, 3};
  bool broadcast = true;
  std::vector<int> output_shape{1, 2, 2, 3};
  int in0_data[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  int in1_data[3] = {-1, 1, 1};
  int out_data[12] = {0};
  schema::PrimitiveType type = schema::PrimitiveType_Mul;
  int act_type = schema::ActivationType_RELU;
  int thread_num = 2;
  desc_.type = type;
  PrepareInt(input0_shape, input1_shape, broadcast, output_shape, in0_data, in1_data, out_data, type, act_type,
             thread_num);
  kernel_->Run();

  int correct_data[12] = {0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11};

  ASSERT_EQ(0, CompareOutputData(out_data, correct_data, 12, err_tol));
}

TEST_F(TestArithmeticTestFp32, MulReluInt1) {
  std::vector<int> input0_shape{1, 2, 2, 3};
  std::vector<int> input1_shape{1};
  bool broadcast = true;
  std::vector<int> output_shape{1, 2, 2, 3};
  int in0_data[12] = {0, -1, -2, -3, -4, -5, 6, 7, 8, 9, 10, 11};
  int in1_data[1] = {1};
  int out_data[12] = {0};
  schema::PrimitiveType type = schema::PrimitiveType_Mul;
  int act_type = schema::ActivationType_RELU;
  int thread_num = 2;
  desc_.type = type;
  PrepareInt(input0_shape, input1_shape, broadcast, output_shape, in0_data, in1_data, out_data, type, act_type,
             thread_num);
  kernel_->Run();

  int correct_data[12] = {0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 11};

  ASSERT_EQ(0, CompareOutputData(out_data, correct_data, 12, err_tol));
}

TEST_F(TestArithmeticTestFp32, MulReluInt2) {
  std::vector<int> input0_shape{1};
  std::vector<int> input1_shape{1, 2, 2, 3};
  bool broadcast = true;
  std::vector<int> output_shape{1, 2, 2, 3};
  int in0_data[1] = {1};
  int in1_data[12] = {0, -1, -2, -3, -4, -5, 6, 7, 8, 9, 10, 11};
  int out_data[12] = {0};
  schema::PrimitiveType type = schema::PrimitiveType_Mul;
  int act_type = schema::ActivationType_RELU;
  int thread_num = 2;
  desc_.type = type;
  PrepareInt(input0_shape, input1_shape, broadcast, output_shape, in0_data, in1_data, out_data, type, act_type,
             thread_num);
  kernel_->Run();

  int correct_data[12] = {0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 11};

  ASSERT_EQ(0, CompareOutputData(out_data, correct_data, 12, err_tol));
}

TEST_F(TestArithmeticTestFp32, MulReluInt3) {
  std::vector<int> input0_shape{1, 2, 2, 3};
  std::vector<int> input1_shape{1, 2, 2, 3};
  bool broadcast = false;
  std::vector<int> output_shape{1, 2, 2, 3};
  int in0_data[12] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  int in1_data[12] = {0, -1, -2, -3, -4, -5, 6, 7, 8, 9, 10, 11};
  int out_data[12] = {0};
  schema::PrimitiveType type = schema::PrimitiveType_Mul;
  int act_type = schema::ActivationType_RELU;
  int thread_num = 2;
  desc_.type = type;
  PrepareInt(input0_shape, input1_shape, broadcast, output_shape, in0_data, in1_data, out_data, type, act_type,
             thread_num);
  kernel_->Run();

  int correct_data[12] = {0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 11};

  ASSERT_EQ(0, CompareOutputData(out_data, correct_data, 12, err_tol));
}

TEST_F(TestArithmeticTestFp32, MulRelu6Int0) {
  std::vector<int> input0_shape{1, 2, 2, 3};
  std::vector<int> input1_shape{1, 1, 1, 3};
  bool broadcast = true;
  std::vector<int> output_shape{1, 2, 2, 3};
  int in0_data[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  int in1_data[3] = {-1, 1, 1};
  int out_data[12] = {0};
  schema::PrimitiveType type = schema::PrimitiveType_Mul;
  int act_type = schema::ActivationType_RELU6;
  int thread_num = 2;
  desc_.type = type;
  PrepareInt(input0_shape, input1_shape, broadcast, output_shape, in0_data, in1_data, out_data, type, act_type,
             thread_num);
  kernel_->Run();

  int correct_data[12] = {0, 1, 2, 0, 4, 5, 0, 6, 6, 0, 6, 6};

  ASSERT_EQ(0, CompareOutputData(out_data, correct_data, 12, err_tol));
}

TEST_F(TestArithmeticTestFp32, MulRelu6Int1) {
  std::vector<int> input0_shape{1, 2, 2, 3};
  std::vector<int> input1_shape{1};
  bool broadcast = true;
  std::vector<int> output_shape{1, 2, 2, 3};
  int in0_data[12] = {0, -1, -2, -3, -4, -5, 6, 7, 8, 9, 10, 11};
  int in1_data[1] = {1};
  int out_data[12] = {0};
  schema::PrimitiveType type = schema::PrimitiveType_Mul;
  int act_type = schema::ActivationType_RELU6;
  int thread_num = 2;
  desc_.type = type;
  PrepareInt(input0_shape, input1_shape, broadcast, output_shape, in0_data, in1_data, out_data, type, act_type,
             thread_num);
  kernel_->Run();

  int correct_data[12] = {0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6};

  ASSERT_EQ(0, CompareOutputData(out_data, correct_data, 12, err_tol));
}

TEST_F(TestArithmeticTestFp32, MulRelu6Int2) {
  std::vector<int> input0_shape{1};
  std::vector<int> input1_shape{1, 2, 2, 3};
  bool broadcast = true;
  std::vector<int> output_shape{1, 2, 2, 3};
  int in0_data[1] = {1};
  int in1_data[12] = {0, -1, -2, -3, -4, -5, 6, 7, 8, 9, 10, 11};
  int out_data[12] = {0};
  schema::PrimitiveType type = schema::PrimitiveType_Mul;
  int act_type = schema::ActivationType_RELU6;
  int thread_num = 2;
  desc_.type = type;
  PrepareInt(input0_shape, input1_shape, broadcast, output_shape, in0_data, in1_data, out_data, type, act_type,
             thread_num);
  kernel_->Run();

  int correct_data[12] = {0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6};

  ASSERT_EQ(0, CompareOutputData(out_data, correct_data, 12, err_tol));
}

TEST_F(TestArithmeticTestFp32, MulRelu6Int3) {
  std::vector<int> input0_shape{1, 2, 2, 3};
  std::vector<int> input1_shape{1, 2, 2, 3};
  bool broadcast = false;
  std::vector<int> output_shape{1, 2, 2, 3};
  int in0_data[12] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  int in1_data[12] = {0, -1, -2, -3, -4, -5, 6, 7, 8, 9, 10, 11};
  int out_data[12] = {0};
  schema::PrimitiveType type = schema::PrimitiveType_Mul;
  int act_type = schema::ActivationType_RELU6;
  int thread_num = 2;
  desc_.type = type;
  PrepareInt(input0_shape, input1_shape, broadcast, output_shape, in0_data, in1_data, out_data, type, act_type,
             thread_num);
  kernel_->Run();

  int correct_data[12] = {0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6};

  ASSERT_EQ(0, CompareOutputData(out_data, correct_data, 12, err_tol));
}

TEST_F(TestArithmeticTestFp32, AddReluFp32) {
  std::vector<lite::Tensor *> inputs_tensor;
  std::vector<lite::Tensor *> outputs_tensor;

  ArithmeticParameter add_param;
  add_param.broadcasting_ = true;
  add_param.op_parameter_.type_ = schema::PrimitiveType_Add;
  add_param.ndim_ = 4;
  add_param.activation_type_ = schema::ActivationType_RELU;
  add_param.in_shape0_[0] = 1;
  add_param.in_shape0_[1] = 2;
  add_param.in_shape0_[2] = 3;
  add_param.in_shape0_[3] = 4;
  add_param.in_shape1_[0] = 1;
  add_param.in_shape1_[1] = 1;
  add_param.in_shape1_[2] = 1;
  add_param.in_shape1_[3] = 4;
  add_param.out_shape_[0] = 1;
  add_param.out_shape_[1] = 2;
  add_param.out_shape_[2] = 3;
  add_param.out_shape_[3] = 4;

  /* 1x2x3x4 NHWC */
  std::vector<float> input0 = {12.216284, 3.3466918, 15.327419,  5.234958,  0.804376,   9.952188,
                               14.727955, -8.080715, 13.71383,   8.055829,  6.5845337,  -9.25232,
                               -4.24519,  11.550042, 9.262012,   1.2780352, 6.7263746,  -3.9301445,
                               3.764492,  -8.602078, -3.3558068, 13.619035, -2.6694393, 3.2008505};
  std::vector<int> input0_shape = {1, 2, 3, 4};
  std::vector<float> input1 = {0.9035316, 0.022212252, 0.3038014, 0.3478275};
  std::vector<int> input1_shape = {1, 1, 1, 4};

  lite::Tensor input0_tensor;
  lite::Tensor input1_tensor;
  input0_tensor.set_data_type(kNumberTypeFloat32);
  input0_tensor.set_data(input0.data());
  input1_tensor.set_data(input1.data());
  input0_tensor.set_shape(input0_shape);
  input1_tensor.set_shape(input1_shape);
  inputs_tensor.push_back(&input0_tensor);
  inputs_tensor.push_back(&input1_tensor);

  std::vector<float> output(24);
  std::vector<int> output_shape = {1, 2, 3, 4};

  lite::Tensor output0_tensor;
  outputs_tensor.push_back(&output0_tensor);
  output0_tensor.set_data(output.data());
  output0_tensor.set_shape(output_shape);

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Eltwise};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  lite::InnerContext ctx;
  ctx.thread_num_ = 3;
  ASSERT_EQ(lite::RET_OK, ctx.Init());
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&add_param), &ctx, desc, nullptr);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor.shape();
  kernel->Run();

  std::vector<float> correct_out = {
    13.119816, 3.368904, 15.631221, 5.5827856, 1.7079077, 9.9744,    15.031756, 0, 14.617362, 8.078041, 6.888335, 0, 0,
    11.572254, 9.565813, 1.6258626, 7.629906,  0,         4.0682936, 0,         0, 13.641247, 0,        3.548678};
  auto correct_out_ptr = correct_out.data();

  ASSERT_EQ(0, CompareOutputData(output.data(), correct_out_ptr, 24, 0.00001));

  input0_tensor.set_data(nullptr);
  input1_tensor.set_data(nullptr);
  output0_tensor.set_data(nullptr);
}

TEST_F(TestArithmeticTestFp32, AddRelu6Fp32) {
  std::vector<lite::Tensor *> inputs_tensor;
  std::vector<lite::Tensor *> outputs_tensor;

  ArithmeticParameter add_param;
  add_param.broadcasting_ = true;
  add_param.op_parameter_.type_ = schema::PrimitiveType_Add;
  add_param.ndim_ = 4;
  add_param.activation_type_ = schema::ActivationType_RELU6;
  add_param.in_shape0_[0] = 1;
  add_param.in_shape0_[1] = 2;
  add_param.in_shape0_[2] = 3;
  add_param.in_shape0_[3] = 4;
  add_param.in_shape1_[0] = 1;
  add_param.in_shape1_[1] = 1;
  add_param.in_shape1_[2] = 1;
  add_param.in_shape1_[3] = 4;
  add_param.out_shape_[0] = 1;
  add_param.out_shape_[1] = 2;
  add_param.out_shape_[2] = 3;
  add_param.out_shape_[3] = 4;

  /* 1x2x3x4 NHWC */
  std::vector<float> input0 = {12.216284, 3.3466918, 15.327419,  5.234958,  0.804376,   9.952188,
                               14.727955, -8.080715, 13.71383,   8.055829,  6.5845337,  -9.25232,
                               -4.24519,  11.550042, 9.262012,   1.2780352, 6.7263746,  -3.9301445,
                               3.764492,  -8.602078, -3.3558068, 13.619035, -2.6694393, 3.2008505};
  std::vector<int> input0_shape = {1, 2, 3, 4};
  std::vector<float> input1 = {0.9035316, 0.022212252, 0.3038014, 0.3478275};
  std::vector<int> input1_shape = {1, 1, 1, 4};

  lite::Tensor input0_tensor;
  lite::Tensor input1_tensor;
  input0_tensor.set_data_type(kNumberTypeFloat32);
  input0_tensor.set_data(input0.data());
  input1_tensor.set_data(input1.data());
  input0_tensor.set_shape(input0_shape);
  input1_tensor.set_shape(input1_shape);
  inputs_tensor.push_back(&input0_tensor);
  inputs_tensor.push_back(&input1_tensor);

  std::vector<float> output(24);
  std::vector<int> output_shape = {1, 2, 3, 4};

  lite::Tensor output0_tensor;
  outputs_tensor.push_back(&output0_tensor);
  output0_tensor.set_data(output.data());
  output0_tensor.set_shape(output_shape);

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Eltwise};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  lite::InnerContext ctx;
  ctx.thread_num_ = 3;
  ASSERT_EQ(lite::RET_OK, ctx.Init());
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&add_param), &ctx, desc, nullptr);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor.shape();
  kernel->Run();

  std::vector<float> correct_out = {6, 3.368904, 6, 5.5827856, 1.7079077, 6, 6,         0, 6, 6, 6, 0,
                                    0, 6,        6, 1.6258626, 6,         0, 4.0682936, 0, 0, 6, 0, 3.548678};
  auto correct_out_ptr = correct_out.data();

  ASSERT_EQ(0, CompareOutputData(output.data(), correct_out_ptr, 24, 0.00001));

  input0_tensor.set_data(nullptr);
  input1_tensor.set_data(nullptr);
  output0_tensor.set_data(nullptr);
}

TEST_F(TestArithmeticTestFp32, DivReluFp32) {
  std::vector<lite::Tensor *> inputs_tensor;
  std::vector<lite::Tensor *> outputs_tensor;

  ArithmeticParameter div_param;
  div_param.broadcasting_ = true;
  div_param.op_parameter_.type_ = schema::PrimitiveType_Div;
  div_param.ndim_ = 4;
  div_param.activation_type_ = schema::ActivationType_RELU;
  div_param.in_shape0_[0] = 1;
  div_param.in_shape0_[1] = 2;
  div_param.in_shape0_[2] = 3;
  div_param.in_shape0_[3] = 4;
  div_param.in_shape1_[0] = 1;
  div_param.in_shape1_[1] = 1;
  div_param.in_shape1_[2] = 1;
  div_param.in_shape1_[3] = 4;
  div_param.out_shape_[0] = 1;
  div_param.out_shape_[1] = 2;
  div_param.out_shape_[2] = 3;
  div_param.out_shape_[3] = 4;

  /* 1x2x3x4 NHWC */
  std::vector<float> input0 = {12.216284, 3.3466918, 15.327419,  5.234958,  0.804376,   9.952188,
                               14.727955, -8.080715, 13.71383,   8.055829,  6.5845337,  -9.25232,
                               -4.24519,  11.550042, 9.262012,   1.2780352, 6.7263746,  -3.9301445,
                               3.764492,  -8.602078, -3.3558068, 13.619035, -2.6694393, 3.2008505};
  std::vector<int> input0_shape = {1, 2, 3, 4};
  std::vector<float> input1 = {1.6771512, -7.336843, 0.6768286, 4.453379};
  std::vector<int> input1_shape = {1, 1, 1, 4};

  lite::Tensor input0_tensor;
  lite::Tensor input1_tensor;
  input0_tensor.set_data_type(kNumberTypeFloat32);
  input0_tensor.set_data(input0.data());
  input1_tensor.set_data(input1.data());
  input0_tensor.set_shape(input0_shape);
  input1_tensor.set_shape(input1_shape);
  inputs_tensor.push_back(&input0_tensor);
  inputs_tensor.push_back(&input1_tensor);

  std::vector<float> output(24);
  std::vector<int> output_shape = {1, 2, 3, 4};

  lite::Tensor output0_tensor;
  outputs_tensor.push_back(&output0_tensor);
  output0_tensor.set_data(output.data());
  output0_tensor.set_shape(output_shape);

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Eltwise};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  lite::InnerContext ctx;
  ctx.thread_num_ = 3;
  ASSERT_EQ(lite::RET_OK, ctx.Init());
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&div_param), &ctx, desc, nullptr);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor.shape();
  kernel->Run();

  std::vector<float> correct_out = {7.28394912,  0, 22.64593872, 1.17550247, 0.47960852, 0,
                                    21.76024329, 0, 8.17685967,  0,          9.72850985, 0,
                                    0,           0, 13.68442764, 0.28698101, 4.01059523, 0.53567243,
                                    5.56195764,  0, 0,           0,          0,          0.71874648};
  auto correct_out_ptr = correct_out.data();

  ASSERT_EQ(0, CompareOutputData(output.data(), correct_out_ptr, 24, 0.00001));

  input0_tensor.set_data(nullptr);
  input1_tensor.set_data(nullptr);
  output0_tensor.set_data(nullptr);
}

TEST_F(TestArithmeticTestFp32, DivRelu6Fp32) {
  std::vector<lite::Tensor *> inputs_tensor;
  std::vector<lite::Tensor *> outputs_tensor;

  ArithmeticParameter div_param;
  div_param.broadcasting_ = true;
  div_param.op_parameter_.type_ = schema::PrimitiveType_Div;
  div_param.ndim_ = 4;
  div_param.activation_type_ = schema::ActivationType_RELU6;
  div_param.in_shape0_[0] = 1;
  div_param.in_shape0_[1] = 2;
  div_param.in_shape0_[2] = 3;
  div_param.in_shape0_[3] = 4;
  div_param.in_shape1_[0] = 1;
  div_param.in_shape1_[1] = 1;
  div_param.in_shape1_[2] = 1;
  div_param.in_shape1_[3] = 4;
  div_param.out_shape_[0] = 1;
  div_param.out_shape_[1] = 2;
  div_param.out_shape_[2] = 3;
  div_param.out_shape_[3] = 4;

  /* 1x2x3x4 NHWC */
  std::vector<float> input0 = {12.216284, 3.3466918, 15.327419,  5.234958,  0.804376,   9.952188,
                               14.727955, -8.080715, 13.71383,   8.055829,  6.5845337,  -9.25232,
                               -4.24519,  11.550042, 9.262012,   1.2780352, 6.7263746,  -3.9301445,
                               3.764492,  -8.602078, -3.3558068, 13.619035, -2.6694393, 3.2008505};
  std::vector<int> input0_shape = {1, 2, 3, 4};
  std::vector<float> input1 = {1.6771512, -7.336843, 0.6768286, 4.453379};
  std::vector<int> input1_shape = {1, 1, 1, 4};

  lite::Tensor input0_tensor;
  lite::Tensor input1_tensor;
  input0_tensor.set_data_type(kNumberTypeFloat32);
  input0_tensor.set_data(input0.data());
  input1_tensor.set_data(input1.data());
  input0_tensor.set_shape(input0_shape);
  input1_tensor.set_shape(input1_shape);
  inputs_tensor.push_back(&input0_tensor);
  inputs_tensor.push_back(&input1_tensor);

  std::vector<float> output(24);
  std::vector<int> output_shape = {1, 2, 3, 4};

  lite::Tensor output0_tensor;
  outputs_tensor.push_back(&output0_tensor);
  output0_tensor.set_data(output.data());
  output0_tensor.set_shape(output_shape);

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Eltwise};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  lite::InnerContext ctx;
  ctx.thread_num_ = 3;
  ASSERT_EQ(lite::RET_OK, ctx.Init());
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&div_param), &ctx, desc, nullptr);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor.shape();
  kernel->Run();

  std::vector<float> correct_out = {6, 0, 6, 1.17550247, 0.47960852, 0,          6,          0, 6, 0, 6, 0,
                                    0, 0, 6, 0.28698101, 4.01059523, 0.53567243, 5.56195764, 0, 0, 0, 0, 0.71874648};
  auto correct_out_ptr = correct_out.data();

  ASSERT_EQ(0, CompareOutputData(output.data(), correct_out_ptr, 24, 0.00001));

  input0_tensor.set_data(nullptr);
  input1_tensor.set_data(nullptr);
  output0_tensor.set_data(nullptr);
}

TEST_F(TestArithmeticTestFp32, EqualFp32) {
  std::vector<lite::Tensor *> inputs_tensor;
  std::vector<lite::Tensor *> outputs_tensor;

  ArithmeticParameter equal_param;
  equal_param.broadcasting_ = true;
  equal_param.op_parameter_.type_ = schema::PrimitiveType_Equal;
  equal_param.ndim_ = 4;
  equal_param.in_shape0_[0] = 1;
  equal_param.in_shape0_[1] = 2;
  equal_param.in_shape0_[2] = 3;
  equal_param.in_shape0_[3] = 4;
  equal_param.in_shape1_[0] = 1;
  equal_param.in_shape1_[1] = 1;
  equal_param.in_shape1_[2] = 1;
  equal_param.in_shape1_[3] = 4;
  equal_param.out_shape_[0] = 1;
  equal_param.out_shape_[1] = 2;
  equal_param.out_shape_[2] = 3;
  equal_param.out_shape_[3] = 4;

  /* 1x2x3x4 NHWC */
  std::vector<float> input0 = {12.216284, 3.3466918, 15.327419,  5.234958,  0.804376,   9.952188,
                               14.727955, -8.080715, 13.71383,   8.055829,  6.5845337,  -9.25232,
                               -4.24519,  11.550042, 9.262012,   1.2780352, 6.7263746,  -3.9301445,
                               3.764492,  -8.602078, -3.3558068, 13.619035, -2.6694393, 3.2008505};
  std::vector<int> input0_shape = {1, 2, 3, 4};
  std::vector<float> input1 = {0.16771512, 3.3466918, 0.6768286, 3.2008505};
  std::vector<int> input1_shape = {1, 1, 1, 4};

  lite::Tensor input0_tensor;
  lite::Tensor input1_tensor;
  input0_tensor.set_data_type(kNumberTypeFloat32);
  input0_tensor.set_data(input0.data());
  input1_tensor.set_data(input1.data());
  input0_tensor.set_shape(input0_shape);
  input1_tensor.set_shape(input1_shape);
  inputs_tensor.push_back(&input0_tensor);
  inputs_tensor.push_back(&input1_tensor);

  std::vector<float> output(24);
  std::vector<int> output_shape = {1, 2, 3, 4};

  lite::Tensor output0_tensor;
  outputs_tensor.push_back(&output0_tensor);
  output0_tensor.set_data(output.data());
  output0_tensor.set_shape(output_shape);

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Eltwise};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  lite::InnerContext ctx;
  ctx.thread_num_ = 3;
  ASSERT_EQ(lite::RET_OK, ctx.Init());
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&equal_param), &ctx, desc, nullptr);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor.shape();
  kernel->Run();

  std::vector<float> correct_out = {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
  auto correct_out_ptr = correct_out.data();

  ASSERT_EQ(0, CompareOutputData(output.data(), correct_out_ptr, 24, 0.00001));

  input0_tensor.set_data(nullptr);
  input1_tensor.set_data(nullptr);
  output0_tensor.set_data(nullptr);
}
}  // namespace mindspore
