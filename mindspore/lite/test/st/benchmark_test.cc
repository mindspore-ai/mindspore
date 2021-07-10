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
#include <gtest/gtest.h>
#include <string>
#include "common/common_test.h"
#include "tools/benchmark/run_benchmark.h"

namespace mindspore {
namespace lite {
class BenchmarkTest : public mindspore::CommonTest {
 public:
  BenchmarkTest() {}
};

TEST_F(BenchmarkTest, TestVideo) {
  const char *argv[] = {"./benchmark", "--modelPath=./hiai/hiai_label_and_video.ms",
                        "--inDataPath=./hiai/hiai_label_and_video.bin",
                        "--calibDataPath=./hiai/hiai_label_and_video.txt"};
  auto status = RunBenchmark(4, argv);
  ASSERT_EQ(status, RET_OK);
}

TEST_F(BenchmarkTest, TestOCR_02) {
  const char *argv[] = {"./benchmark", "--modelPath=./hiai/hiai_cv_focusShootOCRMOdel_02.ms",
                        "--inDataPath=./hiai/hiai_cv_focusShootOCRMOdel_02.bin",
                        "--calibDataPath=./hiai/hiai_cv_focusShootOCRMOdel_02.txt"};
  auto status = RunBenchmark(4, argv);
  ASSERT_EQ(status, RET_OK);
}

TEST_F(BenchmarkTest, TestOCR_02_GPU) {
  const char *argv[] = {"./benchmark", "--modelPath=./hiai/model_02.ms", "--inDataPath=./hiai/model_02_in.bin",
                        "--calibDataPath=./hiai/model_02_out.bin", "--device=GPU"};
  auto status = RunBenchmark(5, argv);
  ASSERT_EQ(status, RET_OK);
}
TEST_F(BenchmarkTest, TestOCR_02_GPU_PERF) {
  const char *argv[] = {"./benchmark", "--modelPath=./hiai/model_02.ms", "--inDataPath=./hiai/model_02_in.bin",
                        "--device=GPU"};
  auto status = RunBenchmark(4, argv);
  ASSERT_EQ(status, RET_OK);
}

TEST_F(BenchmarkTest, Test_MV2_GPU) {
  const char *argv[] = {"./benchmark", "--modelPath=./hiai/mobilenet_v2.ms", "--inDataPath=./hiai/mobilenet_v2_in.bin",
                        "--calibDataPath=./hiai/mobilenet_v2_out.bin", "--device=GPU"};
  auto status = RunBenchmark(5, argv);
  ASSERT_EQ(status, RET_OK);
}

TEST_F(BenchmarkTest, Test_MV2_GPU_PERF) {
  const char *argv[] = {"./benchmark", "--modelPath=./hiai/mobilenet_v2.ms", "--inDataPath=./hiai/mobilenet_v2_in.bin",
                        "--device=GPU"};
  auto status = RunBenchmark(4, argv);
  ASSERT_EQ(status, RET_OK);
}

TEST_F(BenchmarkTest, TestHebing) {
  const char *argv[] = {"./benchmark", "--modelPath=./hiai/model_hebing_3branch.ms",
                        "--inDataPath=./hiai/model_hebing_3branch.bin",
                        "--calibDataPath=./hiai/model_hebing_3branch.txt"};
  auto status = RunBenchmark(4, argv);
  ASSERT_EQ(status, RET_OK);
}

TEST_F(BenchmarkTest, mindrtParallelOffline1) {
  const char *benchmark_argv1[] = {"./benchmark",
                                   "--enableParallel=true",
                                   "--numThreads=3",
                                   "--modelFile=./mindrtParallel/mindrt_parallel_model_split.ms",
                                   "--inDataFile=./mindrtParallel/mindrt_parallel_model.bin",
                                   "--benchmarkDataFile=./mindrtParallel/mindrt_parallel_model.out"};
  int converter_ret = mindspore::lite::RunBenchmark(6, benchmark_argv1);
  ASSERT_EQ(converter_ret, lite::RET_OK);
}

TEST_F(BenchmarkTest, mindrtParallelOffline2) {
  const char *benchmark_argv2[] = {"./benchmark",
                                   "--enableParallel=true",
                                   "--numThreads=4",
                                   "--modelFile=./mindrtParallel/mindrt_parallel_model_split.ms",
                                   "--inDataFile=./mindrtParallel/mindrt_parallel_model.bin",
                                   "--benchmarkDataFile=./mindrtParallel/mindrt_parallel_model.out"};
  int converter_ret = mindspore::lite::RunBenchmark(6, benchmark_argv2);
  ASSERT_EQ(converter_ret, lite::RET_OK);
}

TEST_F(BenchmarkTest, mindrtParallelOffline3) {
  const char *benchmark_argv3[] = {"./benchmark",
                                   "--enableParallel=false",
                                   "--numThreads=1",
                                   "--modelFile=./mindrtParallel/mindrt_parallel_model_split.ms",
                                   "--inDataFile=./mindrtParallel/mindrt_parallel_model.bin",
                                   "--benchmarkDataFile=./mindrtParallel/mindrt_parallel_model.out"};
  int converter_ret = mindspore::lite::RunBenchmark(6, benchmark_argv3);
  ASSERT_EQ(converter_ret, lite::RET_OK);
}
}  // namespace lite
}  // namespace mindspore
