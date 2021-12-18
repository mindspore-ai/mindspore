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

#ifndef MINNIE_BENCHMARK_BENCHMARK_H_
#define MINNIE_BENCHMARK_BENCHMARK_H_

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
#ifndef BENCHMARK_CLIP_JSON
#include <nlohmann/json.hpp>
#endif
#include "tools/benchmark/benchmark_base.h"
#include "include/model.h"
#include "tools/common/flag_parser.h"
#include "src/common/file_utils.h"
#include "src/common/utils.h"
#ifdef ENABLE_OPENGL_TEXTURE
#include "tools/common/opengl_util.h"
#endif
#include "include/lite_session.h"

namespace mindspore::lite {

class MS_API Benchmark : public BenchmarkBase {
 public:
  explicit Benchmark(BenchmarkFlags *flags) : BenchmarkBase(flags) {}

  virtual ~Benchmark();

  int RunBenchmark() override;

 protected:
  // call GenerateRandomData to fill inputTensors
  int GenerateInputData() override;
#ifdef ENABLE_OPENGL_TEXTURE
  int GenerateGLTexture(std::map<std::string, GLuint> *inputGlTexture, std::map<std::string, GLuint> *outputGLTexture);

  int LoadGLTexture();

  int ReadGLTextureFile(std::map<std::string, GLuint> *inputGlTexture, std::map<std::string, GLuint> *outputGLTexture);

  int FillGLTextureToTensor(std::map<std::string, GLuint> *gl_texture, mindspore::tensor::MSTensor *tensor,
                            std::string name, void *data = nullptr);
#endif
  int LoadInput() override;

  int ReadInputFile() override;

  int GetDataTypeByTensorName(const std::string &tensor_name) override;

  int InitContext(const std::shared_ptr<Context> &context);

  int CompareOutput() override;

  int CompareDataGetTotalBiasAndSize(const std::string &name, tensor::MSTensor *tensor, float *total_bias,
                                     int *total_size);

  int InitTimeProfilingCallbackParameter() override;

  int InitPerfProfilingCallbackParameter() override;

  int InitDumpTensorDataCallbackParameter() override;

  int InitPrintTensorDataCallbackParameter() override;

  int PrintInputData();

  int MarkPerformance();

  int MarkAccuracy();

  int CheckInputNames();

  int CompareOutputByCosineDistance(float cosine_distance_threshold);

  int CompareDataGetTotalCosineDistanceAndSize(const std::string &name, tensor::MSTensor *tensor,
                                               float *total_cosine_distance, int *total_size);
  tensor::MSTensor *GetTensorByNodeShape(const std::vector<size_t> &node_shape);
  tensor::MSTensor *GetTensorByNameOrShape(const std::string &node_or_tensor_name, const std::vector<size_t> &dims);

 private:
#ifdef ENABLE_OPENGL_TEXTURE
  mindspore::OpenGL::OpenGLRuntime gl_runtime_;
#endif
  session::LiteSession *session_{nullptr};
  std::vector<mindspore::tensor::MSTensor *> ms_inputs_;
  std::unordered_map<std::string, mindspore::tensor::MSTensor *> ms_outputs_;

  KernelCallBack before_call_back_ = nullptr;
  KernelCallBack after_call_back_ = nullptr;
};

}  // namespace mindspore::lite
#endif  // MINNIE_BENCHMARK_BENCHMARK_H_
