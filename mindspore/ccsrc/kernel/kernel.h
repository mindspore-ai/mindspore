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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_KERNEL_H_
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <set>
#include "nlohmann/json.hpp"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "include/common/utils/utils.h"
#include "ir/tensor.h"
#include "abstract/dshape.h"
#include "utils/log_adapter.h"
#include "runtime/device/executor/dynamic_kernel.h"

#ifdef _MSC_VER
#undef OPAQUE
#endif

namespace mindspore {
enum KernelType : int {
  UNKNOWN_KERNEL_TYPE = 0,
  AKG_KERNEL,
  AICPU_KERNEL,
  RT_KERNEL,
  HCCL_KERNEL,
  TBE_KERNEL,
  HOST_KERNEL,
  CPU_KERNEL,
  GPU_KERNEL,
};

namespace kernel {
// Supported fusion type
enum FusionType {
  CONV = 0,
  ELEMWISE,
  COMMREDUCE,
  SEGMENT,
  OPAQUE,
  BN_UPDATE_GRAD,
  BN_GRAD_REDUCE,
  LAYER_NORM_GRAD,
  L2LOSS_MUL_ADDN,
  PURE_BROADCAST,
  INPLACE,
  MATMUL,
  MATMUL_V2,
  GEMM,
  CONV2D_BACKPROP_INPUT,
  CONV2D_BACKPROP_FILTER,
  CONV3D_BACKPROP_INPUT,
  CONV3D_BACKPROP_FILTER,
  CUBE_LAYER_NORM,
  BN_REDUCE,
  BN_UPDATE,
  SOFTMAX_CROSS_ENTROPY_WITH_LOGITS,
  L2_NORMALIZE,
  SOFTMAX,
  L2_LOSS,
  ASCEND_QUANT,
  ASCEND_DEQUANT,
  ASCEND_ANTI_QUANT,
  STRIDED_READ,
  STRIDED_WRITE,
  ASCEND_DEQUANT_S16,
  ASCEND_REQUANT,
  ASCEND_REQUANT_S16,
  MAX_POOL,
  DEPTHWISECONV,
  CONV3D,
  POOL2D,
  POOL3D,
  READ_SELECT,
  WRITE_SELECT,
  COSINE_EMBEDDING_LOSS,
  DILATION_PATTERN,
  BROAD_CAST,
  BATCH_MATMUL,
  CONFUSION_TRANSPOSE,
  DROPOUT_DOMASKV3D,
  UNKNOWN_FUSION_TYPE = -1,
};

enum OpPattern {
  kCommonPattern = 0,
  kFormatAgnosticPattern = 1,
  kBroadcastPattern = 2,
  kReducePattern = 3,
};

// Backend processor
enum Processor {
  UNKNOWN = -1,
  AICORE = 0,
  AICPU,
  CUDA,
  CPU,
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
  bool has_kernel_list = false;
  uint32_t op_para_size;
  KernelJsonInfo() : block_dim(0), op_para_size(0) {}
};

class KernelPack {
 public:
  KernelPack() : json_(nullptr), kernel_(nullptr) {}
  KernelPack(const KernelPack &) = default;
  KernelJsonInfo kernel_json_info() const;
  bool LoadKernelMeta(const std::string &json_f);
  bool ReadFromJsonFile(const std::string &json_f, const std::string &processor);
  const FlexArray *GetJson() const { return json_; }
  const FlexArray *GetKernel() const { return kernel_; }
  ~KernelPack() {
    if (json_ != nullptr) {
      delete[] json_;
      json_ = nullptr;
    }
    if (kernel_ != nullptr) {
      delete[] kernel_;
      kernel_ = nullptr;
    }
  }

 private:
  bool ReadFromJsonFileHelper(std::ifstream &kernel_bin);
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
  Address() : addr(nullptr), size(0) {}
  Address(void *address_addr, size_t address_size) : addr(address_addr), size(address_size) {}
  void *addr;
  size_t size;
};
using AddressPtr = std::shared_ptr<Address>;
using AddressPtrList = std::vector<AddressPtr>;
using StreamType = void *;
// The memory info of kernel launch.
struct KernelLaunchInfo {
  AddressPtrList inputs_;
  AddressPtrList outputs_;
  AddressPtrList workspaces_;
};

class KernelMod {
 public:
  KernelMod() {}
  explicit KernelMod(const AnfNodePtr &anf_node_ptr) : anf_node_(anf_node_ptr) {}
  virtual ~KernelMod() = default;

  bool Launch(const KernelLaunchInfo &kernel_launch_address, void *stream_ptr) {
    return Launch(kernel_launch_address.inputs_, kernel_launch_address.workspaces_, kernel_launch_address.outputs_,
                  stream_ptr);
  }

  virtual void SetInputSizeList(const std::vector<size_t> &size_list) { input_size_list_ = size_list; }
  virtual void SetOutputSizeList(const std::vector<size_t> &size_list) { output_size_list_ = size_list; }
  virtual void SetWorkspaceSizeList(const std::vector<size_t> &size_list) { workspace_size_list_ = size_list; }
  virtual const std::vector<size_t> &GetInputSizeList() const { return input_size_list_; }
  virtual const std::vector<size_t> &GetOutputSizeList() const { return output_size_list_; }
  virtual const std::vector<size_t> &GetWorkspaceSizeList() const { return workspace_size_list_; }
  virtual bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                      const std::vector<AddressPtr> &outputs, void *stream_ptr) = 0;
  virtual device::DynamicKernelPtr GenDynamicKernel(const CNodePtr &cnode_ptr, void *stream_ptr) { return nullptr; }
  virtual std::vector<size_t> GenParameters() { return {}; }
  virtual void ReleaseResource() {}

  virtual void InferOp() {}
  virtual void InitOp() {}
  virtual void UpdateOp() {}
  void set_unique_name(const std::string &unique_name) { unique_name_ = unique_name; }
  void set_fullname(const std::string &fullname) { fullname_ = fullname; }
  void set_is_monad(bool is_monad) { is_monad_ = is_monad; }
  void set_inputs_addr(const std::vector<AddressPtr> &addr) { inputs_addr_ = addr; }
  void set_workspaces_addr(const std::vector<AddressPtr> &addr) { workspaces_addr_ = addr; }
  void set_outputs_addr(const std::vector<AddressPtr> &addr) { outputs_addr_ = addr; }
  const std::vector<AddressPtr> &GetInputsAddr() const { return inputs_addr_; }
  const std::vector<AddressPtr> &GetWorkSpacesAddr() const { return workspaces_addr_; }
  const std::vector<AddressPtr> &GetOutputsAddr() const { return outputs_addr_; }
  void set_stream(StreamType stream) { stream_ = stream; }
  StreamType stream() const { return stream_; }
  void SetAtomicCleanNodes(const std::vector<CNodePtr> &atomic_clean_node);

 protected:
  void InferShape();
  void GetDepndLists(const CNodePtr &cnode);
  void UpdateOutputSizeList();
  bool NeedSkipExecute(const CNodePtr &cnode);

  std::string kernel_name_;
  std::string unique_name_;
  std::string fullname_;
  bool is_monad_{false};
  StreamType stream_{nullptr};
  AnfNodeWeakPtr anf_node_;
  std::map<uint32_t, tensor::TensorPtr> depend_tensor_map_;
  std::vector<CNodeWeakPtr> atomic_clean_nodes_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  std::set<uint32_t> depend_list_;

 private:
  void InferShapeForNopNode(AnfNodePtr *input_node);
  bool InferShapeForDefiniteOutputNode(const CNodePtr &cnode);

  std::vector<AddressPtr> inputs_addr_;
  std::vector<AddressPtr> workspaces_addr_;
  std::vector<AddressPtr> outputs_addr_;
};
using KernelModPtr = std::shared_ptr<KernelMod>;
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_KERNEL_H_
