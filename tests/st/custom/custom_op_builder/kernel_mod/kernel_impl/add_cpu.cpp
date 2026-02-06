#include <vector>
#include "ops/ops_func_impl/op_func_impl.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "ms_extension/api.h"

namespace my_custom_ops {
using namespace mindspore;
using namespace mindspore::kernel;
using namespace mindspore::ops;

class OPS_API AddCPUOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto output_shape = input_infos[kInputIndex0]->GetShape();
    return {output_shape};
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto input = input_infos[kInputIndex0]->GetType();
    return {input};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class AddCPU : public KernelMod {
 public:
  AddCPU() = default;
  ~AddCPU() = default;

  std::vector<KernelAttr> GetOpSupport() override { return {}; }
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    return KRET_OK;
  }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
    return true;
  }
};
}  // namespace my_custom_ops
