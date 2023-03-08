/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_GRAPH_EXECUTOR_LITERT_FUNC_GRAPH_REUSE_MANAGER_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_GRAPH_EXECUTOR_LITERT_FUNC_GRAPH_REUSE_MANAGER_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <unordered_map>
#include <map>
#include "mindspore/core/base/base.h"
#include "include/api/status.h"
#include "include/backend/kernel_graph.h"
#include "src/common/helper/infer_helpers.h"
namespace mindspore {
struct ModelBufPair {
  void *buf = nullptr;
  size_t buf_size = 0;
};
class FuncGraphReuseManager {
 public:
  static FuncGraphReuseManager *GetInstance();
  ~FuncGraphReuseManager();

  FuncGraphPtr GetSharedFuncGraph(std::map<std::string, std::map<std::string, std::string>> config_info);
  Status StoreFuncGraph(FuncGraphPtr func_graph, std::map<std::string, std::map<std::string, std::string>> config_info);

  std::pair<void *, std::shared_ptr<mindspore::infer::helper::InferHelpers>> GetFbModelBuf(
    size_t *data_size, bool *is_shared_fb_buf, std::map<std::string, std::map<std::string, std::string>> config_info);
  Status StoreFbModelBuf(void *model_buf, size_t data_size,
                         std::shared_ptr<mindspore::infer::helper::InferHelpers> helper,
                         std::map<std::string, std::map<std::string, std::string>> config_info);

  KernelGraphPtr GetKernelGraph(std::map<std::string, std::map<std::string, std::string>> config_info);
  Status StoreKernelGraph(std::map<std::string, std::map<std::string, std::string>> config_info,
                          KernelGraphPtr kernel_graph);

  Status GetInOut(std::map<std::string, std::map<std::string, std::string>> config_info,
                  std::vector<tensor::TensorPtr> *in_tensor, std::vector<tensor::TensorPtr> *out_tensor,
                  std::vector<std::string> *in_name, std::vector<std::string> *out_name);
  Status StoreInOut(std::map<std::string, std::map<std::string, std::string>> config_info,
                    std::vector<tensor::TensorPtr> in_tensor, std::vector<tensor::TensorPtr> out_tensor,
                    std::vector<std::string> in_name, std::vector<std::string> out_name);

  void ReleaseSharedFuncGraph(std::map<std::string, std::map<std::string, std::string>> config_info);

 private:
  FuncGraphReuseManager() = default;

 private:
  // runner id <=> function graph ptr
  // the cached funcgraph is cleared when the model impl is destructed
  std::unordered_map<std::string, FuncGraphPtr> all_func_graphs_;
  std::unordered_map<std::string, ModelBufPair> all_fb_model_buf_;
  std::unordered_map<std::string, KernelGraphPtr> all_kernel_graph_;
  std::unordered_map<std::string, std::shared_ptr<mindspore::infer::helper::InferHelpers>> all_infer_helpers_;
  std::unordered_map<std::string, std::vector<tensor::TensorPtr>> all_in_tensors_;
  std::unordered_map<std::string, std::vector<tensor::TensorPtr>> all_out_tensors_;
  std::unordered_map<std::string, std::vector<std::string>> all_in_names_;
  std::unordered_map<std::string, std::vector<std::string>> all_out_names_;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_GRAPH_EXECUTOR_LITERT_FUNC_GRAPH_REUSE_MANAGER_H_
