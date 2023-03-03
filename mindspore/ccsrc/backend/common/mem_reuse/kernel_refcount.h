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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_KERNEL_REFCOUNT_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_KERNEL_REFCOUNT_H_
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <set>

namespace mindspore {
namespace memreuse {
enum RefCountType { kDynamicRefCount, kStaticRefCount };
enum NodeType { kCommonNode, kCommunicationNode };
enum KernelRefType { kCommon, kRefNodeInput, kRefNodeOutput, kCommNotReuse, kCommReuse, kSummary };
static constexpr int kInitIndex = -1;
class KernelRefCount {
 public:
  uint32_t stream_id_;
  int ref_count_;
  // used by dynamic memory pool, it will be reset by ref_count_ when one minibatch end
  int ref_count_dynamic_use_;
  size_t offset_;
  size_t size_;
  int index_;
  KernelRefType type_;
  // remember to reset offset
  KernelRefCount()
      : stream_id_(0),
        ref_count_(0),
        ref_count_dynamic_use_(0),
        offset_(0),
        size_(0),
        index_(kInitIndex),
        type_(kCommon),
        reftype_(kStaticRefCount) {}
  ~KernelRefCount() = default;
  void SetKernelRefCountInfo(int index, size_t size, RefCountType reftype);
  void set_reftype(RefCountType reftype) { reftype_ = reftype; }
  RefCountType reftype() const { return reftype_; }
  uint32_t stream_id() const { return stream_id_; }

 private:
  RefCountType reftype_;
};
using KernelRefCountPtr = std::shared_ptr<KernelRefCount>;
using KernelRefCountPtrList = std::vector<KernelRefCountPtr>;
// the ptr of every kernel to be key
using KernelKey = void *;
using KernelMap = std::map<KernelKey, std::vector<KernelRefCountPtr>>;

class KernelDef {
 public:
  KernelMap inputs_;
  KernelMap outputs_;
  KernelMap wk_space_;
  NodeType type_ = kCommonNode;
  KernelDef() = default;
  ~KernelDef() = default;
  void set_input_refs(const KernelRefCountPtrList &kernelRefPtrList) { input_refs_ = kernelRefPtrList; }
  void set_output_refs(const KernelRefCountPtrList &kernelRefPtrList) { output_refs_ = kernelRefPtrList; }
  KernelRefCountPtrList input_refs() const { return input_refs_; }
  KernelRefCountPtrList output_refs() const { return output_refs_; }
  std::vector<int> GetInputRefIndexs() const;
  std::vector<int> GetOutputRefIndexs() const;
  std::vector<int> GetWorkspaceRefIndexs() const;
  void set_stream_id(uint32_t stream_id) { stream_id_ = stream_id; }
  uint32_t stream_id() const { return stream_id_; }
  void set_kernel_name(const std::string &kernel_name) { kernel_name_ = kernel_name; }
  std::string kernel_name() const { return kernel_name_; }
  void set_scope_full_name(const std::string &scop_name) { scop_full_name_ = scop_name; }
  std::string scope_full_name() const { return scop_full_name_; }
  void InsertInputKernel(const std::shared_ptr<KernelDef> &input_kernel) { input_kernels_.insert(input_kernel); }
  const std::set<std::shared_ptr<KernelDef>> &input_kernels() const { return input_kernels_; }

 private:
  std::string scop_full_name_;
  std::string kernel_name_;
  uint32_t stream_id_{0};
  KernelRefCountPtrList input_refs_;
  KernelRefCountPtrList output_refs_;
  std::set<std::shared_ptr<KernelDef>> input_kernels_;
};
using KernelDefPtr = std::shared_ptr<KernelDef>;
}  // namespace memreuse
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_KERNEL_REFCOUNT_H_
