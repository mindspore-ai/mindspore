/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_TENSORRT_DELEGATE_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_TENSORRT_DELEGATE_H_
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <map>
#include <optional>
#include "include/api/delegate.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_subgraph.h"
#include "src/extendrt/delegate/parameter_cache/embedding_cache_manager.h"
#include "include/api/kernel.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "include/api/context.h"

namespace mindspore::lite {
class TensorRTDelegate : public Delegate {
 public:
  explicit TensorRTDelegate(mindspore::Context *context, const std::string &cache_model_path, size_t vocab_size,
                            size_t device_cache_size, const std::string &serialize_path,
                            const std::map<std::string, std::string> &input_ranges);
  ~TensorRTDelegate() override;

  Status Init() override;

  Status Build(DelegateModel<schema::Primitive> *model) override;

 private:
  int ParseOptimizationProfile();
  bool ParseOptDims(const std::string &opt_dims_str,
                    const std::unordered_map<std::string, nvinfer1::Dims> &name2input_shape);
  bool ParseDynamicDims(const std::string &dynamic_dims_str,
                        const std::unordered_map<std::string, nvinfer1::Dims> &name2input_shape);
  std::experimental::optional<std::unordered_map<std::string, nvinfer1::Dims>> ParseInputShape(
    const std::string &input_shapes_str);
  Status BuildSubGraph(DelegateModel<schema::Primitive> *model);

  TensorRTOp *FindTensorRTOp(kernel::Kernel *kernel, const schema::Primitive *primitive);

  TensorRTSubGraph *CreateTensorRTGraph(const std::vector<TensorRTOp *> &ops, DelegateModel<schema::Primitive> *model,
                                        KernelIter from, KernelIter end, int index);

  std::unordered_map<schema::PrimitiveType, TensorRTGetOp> op_func_lists_;
  mindspore::Context *context_{nullptr};
  std::shared_ptr<GPUDeviceInfo> device_info_{nullptr};
  TensorRTRuntime *runtime_{nullptr};
  bool support_hw_resize_{true};
  bool support_resize_{true};
  const std::string cache_model_path_;
  size_t vocab_size_{0};
  size_t device_cache_size_{0};
  std::shared_ptr<cache::EmbeddingCacheManager> cache_mgr_{nullptr};
  std::string serialize_path_;
  cudaStream_t stream_{nullptr};
  std::unordered_map<std::string, std::vector<nvinfer1::Dims>> min_dims_;
  std::unordered_map<std::string, std::vector<nvinfer1::Dims>> opt_dims_;
  std::unordered_map<std::string, std::vector<nvinfer1::Dims>> max_dims_;
  std::vector<std::string> input_tensor_names_;
  std::map<std::string, std::string> input_ranges_;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_DELEGATE_
