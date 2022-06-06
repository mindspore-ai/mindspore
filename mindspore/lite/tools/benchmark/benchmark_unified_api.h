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

#ifndef MINDSPORE_LITE_TOOLS_BENCHMARK_BENCHMARK_UNIFIED_API_H_
#define MINDSPORE_LITE_TOOLS_BENCHMARK_BENCHMARK_UNIFIED_API_H_

#include <signal.h>
#include <random>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <map>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <cfloat>
#include <utility>
#include <atomic>
#ifndef BENCHMARK_CLIP_JSON
#include <nlohmann/json.hpp>
#endif
#include "tools/benchmark/benchmark_base.h"
#include "tools/common/flag_parser.h"
#include "src/common/file_utils.h"
#include "src/common/utils.h"
#include "include/api/types.h"
#include "include/api/model.h"
#include "tools/common/opengl_util.h"
#ifdef PARALLEL_INFERENCE
#include "include/api/model_parallel_runner.h"
#endif

namespace mindspore::lite {
class MS_API BenchmarkUnifiedApi : public BenchmarkBase {
 public:
  explicit BenchmarkUnifiedApi(BenchmarkFlags *flags) : BenchmarkBase(flags) {}

  virtual ~BenchmarkUnifiedApi();

  int RunBenchmark() override;

 protected:
  int CompareDataGetTotalBiasAndSize(const std::string &name, mindspore::MSTensor *tensor, float *total_bias,
                                     int *total_size);
  int CompareDataGetTotalCosineDistanceAndSize(const std::string &name, mindspore::MSTensor *tensor,
                                               float *total_cosine_distance, int *total_size);
  void InitContext(const std::shared_ptr<mindspore::Context> &context);

  int CompileGraph(mindspore::ModelType model_type, const std::shared_ptr<Context> &context,
                   const std::string &model_name);

  int GenerateGLTexture(std::map<std::string, GLuint> *inputGlTexture);

  int LoadAndBindGLTexture();

  int ReadGLTextureFile(std::map<std::string, GLuint> *inputGlTexture);

  int FillGLTextureToTensor(std::map<std::string, GLuint> *gl_texture, mindspore::MSTensor *tensor, std::string name,
                            void *data = nullptr);

  // call GenerateRandomData to fill inputTensors
  int LoadInput() override;
  int GenerateInputData() override;

  int ReadInputFile() override;

  int InitMSContext(const std::shared_ptr<Context> &context);

  int GetDataTypeByTensorName(const std::string &tensor_name) override;

  int CompareOutput() override;

  int CompareOutputByCosineDistance(float cosine_distance_threshold);

  int InitTimeProfilingCallbackParameter() override;

  int InitPerfProfilingCallbackParameter() override;

  int InitDumpTensorDataCallbackParameter() override;

  int InitPrintTensorDataCallbackParameter() override;

  int PrintInputData();
#ifdef PARALLEL_INFERENCE
  int CompareOutputForModelPool(std::vector<mindspore::MSTensor> *outputs);
  void ModelParallelRunnerWarmUp(int index);
  void ModelParallelRunnerRun(int task_num, int parallel_idx);
  int ParallelInference(std::shared_ptr<mindspore::Context> context);
  int AddConfigInfo(const std::shared_ptr<RunnerConfig> &runner_config);
#endif

  template <typename T>
  std::vector<int64_t> ConverterToInt64Vector(const std::vector<T> &srcDims) {
    std::vector<int64_t> dims;
    for (auto shape : srcDims) {
      dims.push_back(static_cast<int64_t>(shape));
    }
    return dims;
  }

  int MarkPerformance();

  int MarkAccuracy();

  void UpdateDistributionName(const std::shared_ptr<mindspore::Context> &context, std::string *name);

 private:
  void UpdateConfigInfo();

 private:
  mindspore::OpenGL::OpenGLRuntime gl_runtime_;
  mindspore::Model ms_model_;
  std::vector<mindspore::MSTensor> ms_inputs_for_api_;
  std::vector<mindspore::MSTensor> ms_outputs_for_api_;

  MSKernelCallBack ms_before_call_back_ = nullptr;
  MSKernelCallBack ms_after_call_back_ = nullptr;
#ifdef PARALLEL_INFERENCE
  std::vector<std::vector<int64_t>> resize_dims_;
  std::vector<std::vector<void *>> all_inputs_data_;
  std::vector<std::vector<mindspore::MSTensor>> all_outputs_;
  std::atomic<bool> model_parallel_runner_ret_failed_{false};
  std::atomic<bool> runner_run_start_ = false;
  mindspore::ModelParallelRunner model_runner_;
#endif
};

}  // namespace mindspore::lite
#endif  // MINNIE_BENCHMARK_BENCHMARK_H_
