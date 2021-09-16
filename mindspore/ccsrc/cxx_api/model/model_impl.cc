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
#include "cxx_api/dlutils.h"

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

bool ModelImpl::HasPreprocess() { return graph_->graph_data_->GetPreprocess().empty() ? false : true; }

Status ModelImpl::Preprocess(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
#if !defined(_WIN32) && !defined(_WIN64)
  // Config preprocessor, temporary way to let mindspore.so depends on _c_dataengine
  std::string dataengine_so_path;
  Status dlret = DLSoPath(&dataengine_so_path);
  CHECK_FAIL_AND_RELEASE(dlret, nullptr, "Parse dataengine_so failed: " + dlret.GetErrDescription());

  // Run preprocess
  if (!HasPreprocess()) {
    return Status(kMEFailed, "Attempt to predict with data preprocessor, but no preprocessor is defined in MindIR.");
  }
  std::vector<std::shared_ptr<dataset::Execute>> preprocessor = graph_->graph_data_->GetPreprocess();

  void *handle = nullptr;
  void *function = nullptr;
  dlret = DLSoOpen(dataengine_so_path, "ExecuteRun_C", &handle, &function);
  CHECK_FAIL_AND_RELEASE(dlret, handle, "Parse ExecuteRun_C failed: " + dlret.GetErrDescription());

  auto ExecuteRun =
    (void (*)(const std::vector<std::shared_ptr<dataset::Execute>> &, const std::vector<mindspore::MSTensor> &,
              std::vector<mindspore::MSTensor> *, Status *))(function);
  ExecuteRun(preprocessor, inputs, outputs, &dlret);
  CHECK_FAIL_AND_RELEASE(dlret, handle, "Run preprocess failed: " + dlret.GetErrDescription());
  DLSoClose(handle);
  return kSuccess;
#else
  MS_LOG(ERROR) << "Data preprocess is not supported on Windows yet.";
  return Status(kMEFailed, "Data preprocess is not supported on Windows yet.");
#endif
}

Status ModelImpl::PredictWithPreprocess(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
#if !defined(_WIN32) && !defined(_WIN64)
  // Run preprocess
  std::vector<MSTensor> preprocess_outputs;
  Status ret = Preprocess(inputs, &preprocess_outputs);
  if (ret != kSuccess) {
    return ret;
  }

  // Run prediction
  ret = Predict(preprocess_outputs, outputs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Run predict failed: " << ret.GetErrDescription();
    return ret;
  }
  return kSuccess;
#else
  MS_LOG(ERROR) << "Predict with data preprocess is not supported on Windows yet.";
  return Status(kMEFailed, "Predict with data preprocess is not supported on Windows yet.");
#endif
}
}  // namespace mindspore
