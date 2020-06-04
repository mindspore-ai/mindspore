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
#ifndef MINDSPORE_CCSRC_KERNEL_KERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_KERNEL_H_
#include <vector>
#include <string>
#include <memory>
#include "nlohmann/json.hpp"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "utils/utils.h"
#include "ir/meta_tensor.h"
#include "pipeline/static_analysis/dshape.h"
#include "utils/log_adapter.h"

namespace mindspore {
enum KernelType : int { UNKNOWN_KERNEL_TYPE = 0, AKG_KERNEL, AICPU_KERNEL, RT_KERNEL, HCCL_KERNEL, TBE_KERNEL };

namespace kernel {

enum Axis {
  N = 0,
  C,
  H,
  W,
};

// Supported fusion type
enum FusionType {
  CONVLUTION = 0,
  ELEMWISE,
  COMMREDUCE,
  SEGMENT,
  OPAQUE,
  UNKNOWN_FUSION_TYPE = -1,
};
enum OpPattern {
  kCommonPattern = 0,
  kFormatAgnosticPattern = 1,
  kBroadcastPattern = 2,
  kReducePattern = 3,
  kDynamicFormatPattern = 4,
};

// Backend processor
enum Processor {
  AICORE = 0,
  AICPU,
  CUDA,
};

struct FlexArray {
  size_t len;
  char contents[];
};

struct KernelJsonInfo {
  std::string bin_file_name;
  std::string bin_file_suffix;
  uint32_t block_dim;
  std::string kernel_name;
  std::string magic;
  std::vector<size_t> parameters;
  std::string sha256;
  std::vector<size_t> workspaces;
  KernelJsonInfo() : block_dim(0) {}
};

class KernelPack {
 public:
  KernelPack() : json_(nullptr), kernel_(nullptr) {}
  KernelPack(const KernelPack &) = default;
  KernelJsonInfo kernel_json_info() const;
  bool LoadKernelMeta(const std::string &json_f, const std::string &processor);
  bool ReadFromJsonFile(const std::string &json_f, const std::string &processor);
  const std::string Serialize() const;
  const FlexArray *const GetJson() const { return json_; }
  const FlexArray *const GetKernel() const { return kernel_; }
  ~KernelPack() {
    if (json_) {
      delete[] json_;
      json_ = nullptr;
    }
    if (kernel_) {
      delete[] kernel_;
      kernel_ = nullptr;
    }
  }

 private:
  bool ReadFromJsonFileHelper(std::ifstream &kernelbin);
  void ParseKernelJson(const nlohmann::json &js);
  KernelJsonInfo kernel_json_info_;
  FlexArray *json_;
  FlexArray *kernel_;
};
using KernelPackPtr = std::shared_ptr<KernelPack>;

/**
 * @brief base class for autotensor kernel and cce kernel.
 */
struct Address {
  void *addr;
  size_t size;
};
using AddressPtr = std::shared_ptr<Address>;

class KernelMod {
 public:
  virtual const std::vector<size_t> &GetInputSizeList() const = 0;
  virtual const std::vector<size_t> &GetOutputSizeList() const = 0;
  virtual const std::vector<size_t> &GetWorkspaceSizeList() const = 0;
  virtual bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                      const std::vector<AddressPtr> &outputs, void *stream_ptr) = 0;
  virtual std::vector<size_t> GenParameters() { return {}; }

  virtual ~KernelMod() = default;
};
using KernelModPtr = std::shared_ptr<KernelMod>;
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_KERNEL_H_
