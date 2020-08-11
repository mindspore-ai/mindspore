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
#include <memory>
#include "utils/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/src/runtime/kernel/arm/nnacl/fp32/reduce.h"

namespace mindspore {

class TestReduceFp32 : public mindspore::CommonTest {
 public:
  TestReduceFp32() = default;
  int tid = 0;
  int thread_num = 1;
  float err_tol = 1e-5;
};

TEST_F(TestReduceFp32, Mean) {
  /* 2 4 4 3 NHWC */
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[24] = {18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
                       66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0};

  int input_shape[4] = {2, 4, 4, 3};
  // int output_shape[4] = {2, 1, 4, 3};

  float out[24] = {0};
  int outer_size = 2;
  int inner_size = 12;
  int axis_size = 4;
  (void)ReduceMean(outer_size, inner_size, axis_size, in, input_shape, out, tid, thread_num);

  int output_size = 24;
  CompareOutputData(out, correct, output_size, err_tol);
}

TEST_F(TestReduceFp32, Mean2Thread) {
  /* 2*4*4*3 NHWC */
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[24] = {18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
                       66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0};

  int input_shape[4] = {2, 4, 4, 3};
  // int output_shape[4] = {2, 1, 4, 3};

  float out[24] = {0};
  int outer_size = 2;
  int inner_size = 12;
  int axis_size = 4;
  thread_num = 2;
  tid = 0;
  (void)ReduceMean(outer_size, inner_size, axis_size, in, input_shape, out, tid, thread_num);
  tid = 1;
  (void)ReduceMean(outer_size, inner_size, axis_size, in, input_shape, out, tid, thread_num);

  int output_size = 24;
  CompareOutputData(out, correct, output_size, err_tol);
}

TEST_F(TestReduceFp32, MeanAllAxis) {
  /* 2*4*4*3 NHWC */
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[1] = {47.5};
  float out[1] = {0};

  int input_shape[4] = {2, 4, 4, 3};
  int outer_size = 1;
  int inner_size = 48;
  int axis_size = 2;
  float *src = in;
  float dst1[48] = {0};
  MS_ASSERT(dst != nullptr);
  (void)ReduceMean(outer_size, inner_size, axis_size, src, input_shape, dst1, tid, thread_num);

  input_shape[0] = 1;  // 1 4 4 3
  outer_size = 1;
  inner_size = 12;
  axis_size = 4;
  src = dst1;
  float dst2[12] = {0};
  (void)ReduceMean(outer_size, inner_size, axis_size, src, input_shape, dst2, tid, thread_num);

  input_shape[1] = 1;  // 1 1 4 3
  outer_size = 1;
  inner_size = 3;
  axis_size = 4;
  src = dst2;
  float dst3[3] = {0};
  (void)ReduceMean(outer_size, inner_size, axis_size, src, input_shape, dst3, tid, thread_num);

  input_shape[2] = 1;  // 1 1 1 3
  outer_size = 1;
  inner_size = 1;
  axis_size = 3;
  src = dst3;
  (void)ReduceMean(outer_size, inner_size, axis_size, src, input_shape, out, tid, thread_num);

  int output_size = 1;
  CompareOutputData(out, correct, output_size, err_tol);
}

TEST_F(TestReduceFp32, Sum) {
  /* 2*4*4*3 NHWC */
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[24] = {72.0,  76.0,  80.0,  84.0,  88.0,  92.0,  96.0,  100.0, 104.0, 108.0, 112.0, 116.0,
                       264.0, 268.0, 272.0, 276.0, 280.0, 284.0, 288.0, 292.0, 296.0, 300.0, 304.0, 308.0};

  int input_shape[4] = {2, 4, 4, 3};
  // int output_shape[4] = {2, 1, 4, 3};

  float out[24] = {0};
  int outer_size = 2;
  int inner_size = 12;
  int axis_size = 4;
  (void)ReduceSum(outer_size, inner_size, axis_size, in, input_shape, out, tid, thread_num);

  int output_size = 24;
  CompareOutputData(out, correct, output_size, err_tol);
}

TEST_F(TestReduceFp32, Sum2Thread) {
  /* 2*4*4*3 NHWC */
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[24] = {72.0,  76.0,  80.0,  84.0,  88.0,  92.0,  96.0,  100.0, 104.0, 108.0, 112.0, 116.0,
                       264.0, 268.0, 272.0, 276.0, 280.0, 284.0, 288.0, 292.0, 296.0, 300.0, 304.0, 308.0};

  int input_shape[4] = {2, 4, 4, 3};
  // int output_shape[4] = {2, 1, 4, 3};

  float out[24] = {0};
  int outer_size = 2;
  int inner_size = 12;
  int axis_size = 4;
  thread_num = 2;
  tid = 0;
  (void)ReduceSum(outer_size, inner_size, axis_size, in, input_shape, out, tid, thread_num);
  tid = 1;
  (void)ReduceSum(outer_size, inner_size, axis_size, in, input_shape, out, tid, thread_num);

  int output_size = 24;
  CompareOutputData(out, correct, output_size, err_tol);
}

TEST_F(TestReduceFp32, SumAllAxis) {
  /* 2*4*4*3 NHWC */
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[1] = {4560};
  float out[1] = {0};

  int input_shape[4] = {2, 4, 4, 3};
  int outer_size = 1;
  int inner_size = 48;
  int axis_size = 2;
  float *src = in;
  float dst1[48] = {0};
  MS_ASSERT(dst != nullptr);
  (void)ReduceSum(outer_size, inner_size, axis_size, src, input_shape, dst1, tid, thread_num);

  input_shape[0] = 1;  // 1 4 4 3
  outer_size = 1;
  inner_size = 12;
  axis_size = 4;
  src = dst1;
  float dst2[12] = {0};
  (void)ReduceSum(outer_size, inner_size, axis_size, src, input_shape, dst2, tid, thread_num);

  input_shape[1] = 1;  // 1 1 4 3
  outer_size = 1;
  inner_size = 3;
  axis_size = 4;
  src = dst2;
  float dst3[3] = {0};
  (void)ReduceSum(outer_size, inner_size, axis_size, src, input_shape, dst3, tid, thread_num);

  input_shape[2] = 1;  // 1 1 1 3
  outer_size = 1;
  inner_size = 1;
  axis_size = 3;
  src = dst3;
  (void)ReduceSum(outer_size, inner_size, axis_size, src, input_shape, out, tid, thread_num);

  int output_size = 1;
  CompareOutputData(out, correct, output_size, err_tol);
}

TEST_F(TestReduceFp32, Max) {
  /* 2*4*4*3 NHWC */
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[24] = {36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                       84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};

  int input_shape[4] = {2, 4, 4, 3};
  // int output_shape[4] = {2, 1, 4, 3};

  float out[24] = {0};
  int outer_size = 2;
  int inner_size = 12;
  int axis_size = 4;
  (void)ReduceMax(outer_size, inner_size, axis_size, in, input_shape, out, tid, thread_num);

  int output_size = 24;
  CompareOutputData(out, correct, output_size, err_tol);
}

TEST_F(TestReduceFp32, Min) {
  /* 2*4*4*3 NHWC */
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[24] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0,
                       48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0};

  int input_shape[4] = {2, 4, 4, 3};
  // int output_shape[4] = {2, 1, 4, 3};

  float out[24] = {0};
  int outer_size = 2;
  int inner_size = 12;
  int axis_size = 4;
  (void)ReduceMin(outer_size, inner_size, axis_size, in, input_shape, out, tid, thread_num);

  int output_size = 24;
  CompareOutputData(out, correct, output_size, err_tol);
}

TEST_F(TestReduceFp32, Prod) {
  /* 2*4*4*3 NHWC */
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[24] = {0.0,        12025.0,    27664.0,    47385.0,    71680.0,    101065.0,   136080.0,   177289.0,
                       225280.0,   280665.0,   344080.0,   416185.0,   17418240.0, 18546744.0, 19728400.0, 20964824.0,
                       22257664.0, 23608584.0, 25019280.0, 26491464.0, 28026880.0, 29627288.0, 31294480.0, 33030264.0};

  int input_shape[4] = {2, 4, 4, 3};
  // int output_shape[4] = {2, 1, 4, 3};

  float out[24] = {0};
  int outer_size = 2;
  int inner_size = 12;
  int axis_size = 4;
  (void)ReduceProd(outer_size, inner_size, axis_size, in, input_shape, out, tid, thread_num);

  int output_size = 24;
  CompareOutputData(out, correct, output_size, err_tol);
}

TEST_F(TestReduceFp32, SumSquare) {
  /* 2*4*4*3 NHWC */
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[24] = {2016.0,  2164.0,  2320.0,  2484.0,  2656.0,  2836.0,  3024.0,  3220.0,
                       3424.0,  3636.0,  3856.0,  4084.0,  18144.0, 18676.0, 19216.0, 19764.0,
                       20320.0, 20884.0, 21456.0, 22036.0, 22624.0, 23220.0, 23824.0, 24436.0};

  int input_shape[4] = {2, 4, 4, 3};
  // int output_shape[4] = {2, 1, 4, 3};

  float out[24] = {0};
  int outer_size = 2;
  int inner_size = 12;
  int axis_size = 4;
  (void)ReduceSumSquare(outer_size, inner_size, axis_size, in, input_shape, out, tid, thread_num);

  int output_size = 24;
  CompareOutputData(out, correct, output_size, err_tol);
}
}  // namespace mindspore
