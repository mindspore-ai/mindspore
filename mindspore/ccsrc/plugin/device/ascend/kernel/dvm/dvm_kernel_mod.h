/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_DVM_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_DVM_KERNEL_MOD_H_

#include <memory>
#include <vector>
#include <string>
#include "kernel/kernel.h"
#include "plugin/device/ascend/kernel/dvm/dvm.h"
#include "backend/common/optimizer/dynamic_shape/dynamic_shape_helper.h"

namespace mindspore {
namespace kernel {
using ShapeRefPtr = std::shared_ptr<dvm::ShapeRef>;
using opt::dynamic_shape::InferShapeFunctor;
class DvmKernelMod;
class DvmInfer : public InferShapeFunctor {
 public:
  DvmInfer(const std::string &name, DvmKernelMod *kernel) : InferShapeFunctor(name) { kernel_ = kernel; }
  ~DvmInfer() override = default;
  MS_DECLARE_PARENT(DvmInfer, InferShapeFunctor)
  BaseShapePtr InferShape(const AbstractBasePtrList &args) override;

 private:
  DvmKernelMod *kernel_;
};

class DvmKernelMod : public KernelMod {
 public:
  explicit DvmKernelMod(dvm::KernelType kernel_type);
  ~DvmKernelMod() = default;

  std::vector<KernelAttr> GetOpSupport() override { MS_LOG(EXCEPTION) << "This interface is not support in VKernel."; }

  bool Init(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &) override { return true; }

  int Resize(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &) override { return 0; }

  virtual void Initialize(const std::vector<TypeId> &inputs_type, const std::vector<TypeId> &outputs_type);

  // used in static shape
  void CodeGen(const std::vector<ShapeVector> &inputs_shape, const std::vector<ShapeVector> &outputs_shape);

  virtual void UpdateOutputShapes() = 0;

  size_t GetOutputNum() { return outputs_addr_.size(); }

  // used in dynamic shape
  BaseShapePtr InferShape(const AbstractBasePtrList &inputs_abs);

  dvm::Kernel *Kernel() { return &kernel_; }

  void CacheShapeRef(const ShapeRefPtr &shape_ref) { shapes_ref_.push_back(shape_ref); }

  virtual void UpdateIO() = 0;

  void UpdateInputShapeRef(size_t input_idx, dvm::ShapeRef *ref);

 protected:
  std::vector<ShapeVector> inputs_shape_;
  std::vector<ShapeVector> outputs_shape_;
  std::vector<ShapeRefPtr> shapes_ref_;            // manage the dynamically allocated ShapeRef
  std::vector<dvm::ShapeRef *> inputs_shape_ref_;  // point to the latest inputs shape, which is used in infer shape
  std::vector<void *> inputs_addr_;
  std::vector<void *> outputs_addr_;
  dvm::RelocTable reloc_table_;
  std::vector<size_t> inputs_type_byte_;
  std::vector<size_t> outputs_type_byte_;
  dvm::Kernel kernel_;
};

class SingleDvmKernelMod : public DvmKernelMod {
 public:
  explicit SingleDvmKernelMod(dvm::KernelType kernel_type) : DvmKernelMod(kernel_type) {}
  ~SingleDvmKernelMod() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

  void Initialize(const std::vector<TypeId> &inputs_type, const std::vector<TypeId> &outputs_type) override;

  std::vector<ShapeVector> *ShapesSource() { return &shapes_ref_source_; }

  void CacheLoad(dvm::NDObject *obj, size_t idx);

  void CacheStore(dvm::NDObject *obj, size_t idx);

  void UpdateIO() override;

  void UpdateOutputShapes() override;

 private:
  std::vector<ShapeVector> shapes_ref_source_;  // to ensure the shape which is pointed by ShapeRef keeps alive
  std::vector<dvm::NDObject *> inputs_;         // cache Load
  std::vector<dvm::NDObject *> outputs_;        // cache Store
  std::vector<size_t> inputs_idx_;
  std::vector<size_t> outputs_idx_;
};

class ParallelDvmKernelMod : public DvmKernelMod {
 public:
  ParallelDvmKernelMod(dvm::KernelType kernel_type, size_t sub_graph_count)
      : DvmKernelMod(kernel_type),
        sub_graph_count_(sub_graph_count),
        shapes_ref_source_(sub_graph_count),
        inputs_(sub_graph_count),
        outputs_(sub_graph_count),
        inputs_idx_(sub_graph_count),
        outputs_idx_(sub_graph_count) {}
  ~ParallelDvmKernelMod() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

  std::vector<ShapeVector> *ShapesSource(size_t graph_idx) { return &shapes_ref_source_[graph_idx]; }

  void Initialize(const std::vector<TypeId> &inputs_type, const std::vector<TypeId> &outputs_type) override;

  void CacheLoad(dvm::NDObject *obj, size_t graph_idx, size_t idx);

  void CacheStore(dvm::NDObject *obj, size_t graph_idx, size_t idx);

  void UpdateIO() override;

  void UpdateOutputShapes() override;

 private:
  size_t sub_graph_count_;
  // to ensure the shape which is pointed by ShapeRef keeps alive
  std::vector<std::vector<ShapeVector>> shapes_ref_source_;
  std::vector<std::vector<dvm::NDObject *>> inputs_;
  std::vector<std::vector<dvm::NDObject *>> outputs_;
  std::vector<std::vector<size_t>> inputs_idx_;
  std::vector<std::vector<size_t>> outputs_idx_;
  std::vector<dvm::NDObject *> all_inputs_;
  std::vector<dvm::NDObject *> all_outputs_;
  // map indices of all_inputs_ to input indices of function graph
  std::vector<size_t> inputs_map_;
  // map indices of all_outputs_ to output indices of function graph
  std::vector<size_t> outputs_map_;
};

using DvmKernelModPtr = std::shared_ptr<DvmKernelMod>;
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_DVM_KERNEL_MOD_H_
