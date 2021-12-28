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
#ifndef MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_TENSORRT_DELEGATE_H_
#define MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_TENSORRT_DELEGATE_H_
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include "include/api/delegate.h"
#include "src/delegate/tensorrt/tensorrt_subgraph.h"
#include "src/delegate/parameter_cache/embedding_cache_manager.h"
#include "include/api/kernel.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "include/api/context.h"

namespace mindspore::lite {
typedef TensorRTOp *(*TensorRTGetOp)(const schema::Primitive *primitive,
                                     const std::vector<mindspore::MSTensor> &in_tensors,
                                     const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name);

class TensorRTDelegate : public Delegate {
 public:
  explicit TensorRTDelegate(mindspore::Context *context, const std::string &cache_model_path, size_t vocab_size,
                            size_t device_cache_size, const std::string &serialize_path)
      : context_(context),
        cache_model_path_(cache_model_path),
        vocab_size_(vocab_size),
        device_cache_size_(device_cache_size),
        serialize_path_(serialize_path) {}

  ~TensorRTDelegate() override;

  Status Init() override;

  Status Build(DelegateModel<schema::Primitive> *model) override;

 private:
  Status BuildSubGraph(DelegateModel<schema::Primitive> *model);

  TensorRTOp *FindTensorRTOp(kernel::Kernel *kernel, const schema::Primitive *primitive);

  TensorRTSubGraph *CreateTensorRTGraph(const std::vector<TensorRTOp *> &ops, DelegateModel<schema::Primitive> *model,
                                        KernelIter from, KernelIter end, int index);

  std::unordered_map<schema::PrimitiveType, TensorRTGetOp> op_func_lists_;

  mindspore::Context *context_;

  std::shared_ptr<GPUDeviceInfo> device_info_{nullptr};

  TensorRTRuntime *runtime_{nullptr};

  bool support_hw_resize_{true};

  bool support_resize_{true};
  const std::string cache_model_path_;
  size_t vocab_size_;
  size_t device_cache_size_;
  std::shared_ptr<cache::EmbeddingCacheManager> cache_mgr_{nullptr};
  const std::string serialize_path_;

  cudaStream_t stream_;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_DELEGATE_
