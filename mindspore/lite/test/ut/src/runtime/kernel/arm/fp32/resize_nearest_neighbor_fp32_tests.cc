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
#include "mindspore/lite/src/runtime/kernel/arm/nnacl/fp32/resize.h"

namespace mindspore {

class TestResizeNearestNeighborFp32 : public mindspore::Common {
 public:
  TestResizeNearestNeighborFp32() = default;

 public:
  int tid = 0;
  int thread_num = 1;
  float err_tol = 1e-5;
};

// 1*1 -> 1*1
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest1) {
  std::vector<float> input = {1.0};
  std::vector<int> input_shape = {1, 1, 1, 1};
  std::vector<int> output_shape = {1, 1, 1, 1};
  std::vector<float> expect = {1.0};
  size_t output_size = 1;
  std::vector<float> output(output_size, 0.0);

  ResizeNearestNeighbor(input.data(), output.data(), input_shape.data(), output_shape.data(), tid, thread_num);
  CompareOutputData(output.data(), expect.data(), output_size, err_tol);
}

// 2*2 -> 1*1
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest2) {
  std::vector<float> input = {0.0, 1.0, 2.0, 3.0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 1, 1, 1};
  std::vector<float> expect = {0.0};
  size_t output_size = 1;
  std::vector<float> output(output_size, 0.0);

  ResizeNearestNeighbor(input.data(), output.data(), input_shape.data(), output_shape.data(), tid, thread_num);
  CompareOutputData(output.data(), expect.data(), output_size, err_tol);
}

// 2*2 -> 1*2
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest3) {
  std::vector<float> input = {0.0, 1.0, 2.0, 3.0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 1, 2, 1};
  std::vector<float> expect = {0.0, 1.0};
  size_t output_size = 2;
  std::vector<float> output(output_size, 0.0);

  ResizeNearestNeighbor(input.data(), output.data(), input_shape.data(), output_shape.data(), tid, thread_num);
  CompareOutputData(output.data(), expect.data(), output_size, err_tol);
}

// 2*2 -> 2*1
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest4) {
  std::vector<float> input = {0.0, 1.0, 2.0, 3.0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 2, 1, 1};
  std::vector<float> expect = {0.0, 2.0};
  size_t output_size = 2;
  std::vector<float> output(output_size, 0.0);

  ResizeNearestNeighbor(input.data(), output.data(), input_shape.data(), output_shape.data(), tid, thread_num);
  CompareOutputData(output.data(), expect.data(), output_size, err_tol);
}

// 2*2 -> 2*2
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest5) {
  std::vector<float> input = {0.0, 1.0, 2.0, 3.0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 2, 2, 1};
  std::vector<float> expect = {0.0, 1.0, 2.0, 3.0};
  size_t output_size = 4;
  std::vector<float> output(output_size, 0.0);

  ResizeNearestNeighbor(input.data(), output.data(), input_shape.data(), output_shape.data(), tid, thread_num);
  CompareOutputData(output.data(), expect.data(), output_size, err_tol);
}

// 2*2 -> 1*4
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest6) {
  std::vector<float> input = {0.0, 1.0, 2.0, 3.0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 1, 4, 1};
  std::vector<float> expect = {0.0, 0.0, 1.0, 1.0};
  size_t output_size = 4;
  std::vector<float> output(output_size, 0.0);

  ResizeNearestNeighbor(input.data(), output.data(), input_shape.data(), output_shape.data(), tid, thread_num);
  CompareOutputData(output.data(), expect.data(), output_size, err_tol);
}

// 2*2 -> 4*1
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest7) {
  std::vector<float> input = {0.0, 1.0, 2.0, 3.0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 4, 1, 1};
  std::vector<float> expect = {0.0, 0.0, 2.0, 2.0};
  size_t output_size = 4;
  std::vector<float> output(output_size, 0.0);

  ResizeNearestNeighbor(input.data(), output.data(), input_shape.data(), output_shape.data(), tid, thread_num);
  CompareOutputData(output.data(), expect.data(), output_size, err_tol);
}

// 2*2 -> 2*4
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest8) {
  std::vector<float> input = {0.0, 1.0, 2.0, 3.0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 2, 4, 1};
  std::vector<float> expect = {0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0};
  size_t output_size = 8;
  std::vector<float> output(output_size, 0.0);

  ResizeNearestNeighbor(input.data(), output.data(), input_shape.data(), output_shape.data(), tid, thread_num);
  CompareOutputData(output.data(), expect.data(), output_size, err_tol);
}

// 2*2 -> 4*2
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest9) {
  std::vector<float> input = {0.0, 1.0, 2.0, 3.0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 4, 2, 1};
  std::vector<float> expect = {0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 3.0};
  size_t output_size = 8;
  std::vector<float> output(output_size, 0.0);

  ResizeNearestNeighbor(input.data(), output.data(), input_shape.data(), output_shape.data(), tid, thread_num);
  CompareOutputData(output.data(), expect.data(), output_size, err_tol);
}

// 2*2 -> 3*3
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest10) {
  std::vector<float> input = {0.0, 1.0, 2.0, 3.0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 3, 3, 1};
  std::vector<float> expect = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 3.0};
  size_t output_size = 9;
  std::vector<float> output(output_size, 0.0);

  ResizeNearestNeighbor(input.data(), output.data(), input_shape.data(), output_shape.data(), tid, thread_num);
  CompareOutputData(output.data(), expect.data(), output_size, err_tol);
}

// 2*2 -> 4*4
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest11) {
  std::vector<float> input = {0.0, 1.0, 2.0, 3.0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 4, 4, 1};
  std::vector<float> expect = {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 2.0, 2.0, 3.0, 3.0};
  size_t output_size = 16;
  std::vector<float> output(output_size, 0.0);

  ResizeNearestNeighbor(input.data(), output.data(), input_shape.data(), output_shape.data(), tid, thread_num);
  CompareOutputData(output.data(), expect.data(), output_size, err_tol);
}

// 2*2*2*5 -> 2*4*4*5
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest12) {
  std::vector<float> input = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0,
                              14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0,
                              28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0};
  std::vector<int> input_shape = {2, 2, 2, 5};
  std::vector<int> output_shape = {2, 4, 4, 5};
  std::vector<float> expect = {
    0.0,  1.0,  2.0,  3.0,  4.0,  0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  5.0,  6.0,  7.0,
    8.0,  9.0,  0.0,  1.0,  2.0,  3.0,  4.0,  0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  5.0,
    6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
    19.0, 15.0, 16.0, 17.0, 18.0, 19.0, 10.0, 11.0, 12.0, 13.0, 14.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    17.0, 18.0, 19.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 20.0, 21.0, 22.0, 23.0, 24.0,
    25.0, 26.0, 27.0, 28.0, 29.0, 25.0, 26.0, 27.0, 28.0, 29.0, 20.0, 21.0, 22.0, 23.0, 24.0, 20.0, 21.0, 22.0,
    23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 30.0,
    31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 35.0, 36.0, 37.0, 38.0, 39.0, 30.0, 31.0, 32.0, 33.0,
    34.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 35.0, 36.0, 37.0, 38.0, 39.0};
  size_t output_size = 160;
  std::vector<float> output(output_size, 0.0);

  ResizeNearestNeighbor(input.data(), output.data(), input_shape.data(), output_shape.data(), tid, thread_num);
  CompareOutputData(output.data(), expect.data(), output_size, err_tol);
}

// 2*2*2*5 -> 2*4*4*5 thread_num 2
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest13) {
  std::vector<float> input = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0,
                              14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0,
                              28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0};
  std::vector<int> input_shape = {2, 2, 2, 5};
  std::vector<int> output_shape = {2, 4, 4, 5};
  std::vector<float> expect = {
    0.0,  1.0,  2.0,  3.0,  4.0,  0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  5.0,  6.0,  7.0,
    8.0,  9.0,  0.0,  1.0,  2.0,  3.0,  4.0,  0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  5.0,
    6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
    19.0, 15.0, 16.0, 17.0, 18.0, 19.0, 10.0, 11.0, 12.0, 13.0, 14.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    17.0, 18.0, 19.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 20.0, 21.0, 22.0, 23.0, 24.0,
    25.0, 26.0, 27.0, 28.0, 29.0, 25.0, 26.0, 27.0, 28.0, 29.0, 20.0, 21.0, 22.0, 23.0, 24.0, 20.0, 21.0, 22.0,
    23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 30.0,
    31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 35.0, 36.0, 37.0, 38.0, 39.0, 30.0, 31.0, 32.0, 33.0,
    34.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 35.0, 36.0, 37.0, 38.0, 39.0};
  size_t output_size = 160;
  std::vector<float> output(output_size, 0.0);

  thread_num = 2;
  tid = 0;
  ResizeNearestNeighbor(input.data(), output.data(), input_shape.data(), output_shape.data(), tid, thread_num);
  tid = 1;
  ResizeNearestNeighbor(input.data(), output.data(), input_shape.data(), output_shape.data(), tid, thread_num);
  CompareOutputData(output.data(), expect.data(), output_size, err_tol);
}

// 2*2*2*5 -> 2*4*4*5 thread_num 4
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest14) {
  std::vector<float> input = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0,
                              14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0,
                              28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0};
  std::vector<int> input_shape = {2, 2, 2, 5};
  std::vector<int> output_shape = {2, 4, 4, 5};
  std::vector<float> expect = {
    0.0,  1.0,  2.0,  3.0,  4.0,  0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  5.0,  6.0,  7.0,
    8.0,  9.0,  0.0,  1.0,  2.0,  3.0,  4.0,  0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  5.0,
    6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
    19.0, 15.0, 16.0, 17.0, 18.0, 19.0, 10.0, 11.0, 12.0, 13.0, 14.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    17.0, 18.0, 19.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 20.0, 21.0, 22.0, 23.0, 24.0,
    25.0, 26.0, 27.0, 28.0, 29.0, 25.0, 26.0, 27.0, 28.0, 29.0, 20.0, 21.0, 22.0, 23.0, 24.0, 20.0, 21.0, 22.0,
    23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 30.0,
    31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 35.0, 36.0, 37.0, 38.0, 39.0, 30.0, 31.0, 32.0, 33.0,
    34.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 35.0, 36.0, 37.0, 38.0, 39.0};
  size_t output_size = 160;
  std::vector<float> output(output_size, 0.0);

  thread_num = 4;
  tid = 0;
  ResizeNearestNeighbor(input.data(), output.data(), input_shape.data(), output_shape.data(), tid, thread_num);
  tid = 1;
  ResizeNearestNeighbor(input.data(), output.data(), input_shape.data(), output_shape.data(), tid, thread_num);
  tid = 2;
  ResizeNearestNeighbor(input.data(), output.data(), input_shape.data(), output_shape.data(), tid, thread_num);
  tid = 3;
  ResizeNearestNeighbor(input.data(), output.data(), input_shape.data(), output_shape.data(), tid, thread_num);
  CompareOutputData(output.data(), expect.data(), output_size, err_tol);
}
}  // namespace mindspore
