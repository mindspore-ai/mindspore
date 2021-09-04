/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "cxx_api/model/model_impl.h"

#include <fstream>
#include "debug/common.h"

namespace mindspore {
Status ModelImpl::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  if (graph_ == nullptr) {
    MS_LOG(ERROR) << "Invalid data, graph_ is null.";
    return kMCFailed;
  }

  if (graph_cell_ == nullptr) {
    MS_LOG(WARNING) << "Model has not been built, it will be built with default options";
    Status ret = Build();
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Build model failed.";
      return ret;
    }
  }

  MS_EXCEPTION_IF_NULL(graph_cell_);
  Status ret = graph_cell_->Run(inputs, outputs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Run graph failed.";
    return ret;
  }

  return kSuccess;
}

Status ModelImpl::Predict(const std::string &input, std::vector<MSTensor> *outputs) {
#if !defined(_WIN32) && !defined(_WIN64)
  auto realpath = Common::GetRealPath(input);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed, path=" << input;
    return Status(kMEInvalidInput, "Get real path failed, path=" + input);
  }
  MS_EXCEPTION_IF_NULL(outputs);

  // Read image file
  auto file = realpath.value();
  if (file.empty()) {
    return Status(kMEInvalidInput, "can not find any input file.");
  }

  std::ifstream ifs(file, std::ios::in | std::ios::binary);
  if (!ifs.good()) {
    return Status(kMEInvalidInput, "File: " + file + " does not exist.");
  }
  if (!ifs.is_open()) {
    return Status(kMEInvalidInput, "File: " + file + " open failed.");
  }

  auto &io_seekg1 = ifs.seekg(0, std::ios::end);
  if (!io_seekg1.good() || io_seekg1.fail() || io_seekg1.bad()) {
    ifs.close();
    return Status(kMEInvalidInput, "Failed to seekg file: " + file);
  }

  size_t size = ifs.tellg();
  MSTensor buffer(file, mindspore::DataType::kNumberTypeUInt8, {static_cast<int64_t>(size)}, nullptr, size);

  auto &io_seekg2 = ifs.seekg(0, std::ios::beg);
  if (!io_seekg2.good() || io_seekg2.fail() || io_seekg2.bad()) {
    ifs.close();
    return Status(kMEInvalidInput, "Failed to seekg file: " + file);
  }

  auto &io_read = ifs.read(reinterpret_cast<char *>(buffer.MutableData()), size);
  if (!io_read.good() || io_read.fail() || io_read.bad()) {
    ifs.close();
    return Status(kMEInvalidInput, "Failed to read file: " + file);
  }
  ifs.close();

  // Run preprocess
  std::vector<MSTensor> transform_inputs;
  std::vector<MSTensor> transform_outputs;
  transform_inputs.emplace_back(std::move(buffer));
  MS_LOG(DEBUG) << "transform_inputs[0].Shape: " << transform_inputs[0].Shape();
  auto preprocessor = graph_->graph_data_->GetPreprocess();
  if (!preprocessor.empty()) {
    for (auto exes : preprocessor) {
      MS_EXCEPTION_IF_NULL(exes);
      Status ret = exes->operator()(transform_inputs, &transform_outputs);
      if (ret != kSuccess) {
        MS_LOG(ERROR) << "Run preprocess failed.";
        return ret;
      }
      MS_LOG(DEBUG) << "transform_outputs[0].Shape: " << transform_outputs[0].Shape();
      transform_inputs = transform_outputs;
    }
  } else {
    std::string msg = "Attempt to predict with data preprocess, but no preprocess operation is defined in MindIR.";
    MS_LOG(ERROR) << msg;
    return Status(kMEFailed, msg);
  }

  // Run prediction
  Status ret = Predict(transform_outputs, outputs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << ret.GetErrDescription();
    return ret;
  }
  return kSuccess;
#else
  MS_LOG(ERROR) << "Predict with data preprocess is not supported on Windows yet.";
  return Status(kMEFailed, "Predict with data preprocess is not supported on Windows yet.");
#endif
}
}  // namespace mindspore
