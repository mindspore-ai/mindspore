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

#include "predict/predict.h"

#include <memory>
#include <vector>
#include <string>

namespace mindspore {
namespace predictmodel {
void StepConvertGraph(const KernelGraphPtr &kernel_graph_ptr) {
  MS_LOG(INFO) << "start convert_graph step";
  // get kernel_graph. this graph can be origin or device, depends on which steps to persistence
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  bool save_ms_model = MsContext::GetInstance()->save_ms_model_flag();
  if (save_ms_model) {
    // set convert_mode: convert cpu info or convert Davnici
    executor::Kernel2Ms::GetInstance().set_convert_mode(executor::kConvertCpuMode);
    // convert kernel_graph to sub_ms_graph
    bool ret = executor::Kernel2Ms::GetInstance().KernelGraph2MsGraph(kernel_graph_ptr);
    if (!ret) {
      MS_LOG(WARNING) << "convert to mindsporeGraph failed";
    } else {
      MS_LOG(INFO) << "convert to Graph success";
    }
  }
}

void StepConvertWeight(const std::vector<tensor::TensorPtr> &inputs) {
  MS_LOG(INFO) << "start convert_input step";
  // get all inputs tensor
  bool save_ms_model = MsContext::GetInstance()->save_ms_model_flag();
  std::string save_path = MsContext::GetInstance()->save_ms_model_path();
  if (save_ms_model) {
    MS_LOG(INFO) << "save ms model is true to path " << save_path;
    if (!executor::Kernel2Ms::GetInstance().KernelInput2MS(inputs)) {
      MS_LOG(WARNING) << "convert mindspore kernel input failed";
    }
    auto new_ms_graph_ptr = std::make_shared<mindspore::predict::GraphDefT>();
    bool ret = executor::Kernel2Ms::GetInstance().SaveDeviceModel(new_ms_graph_ptr, save_path);
    if (!ret) {
      MS_LOG(WARNING) << "convert to mindsporeGraph failed";
    } else {
      MS_LOG(INFO) << "save ms model success";
    }
  }
}
}  // namespace predictmodel
}  // namespace mindspore
