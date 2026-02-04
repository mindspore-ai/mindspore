#include <vector>
#include <map>
#include <string>
#include <utility>
#include "ops/ops_func_impl/op_func_impl.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"
#include "ms_extension/api.h"
#include "module.h"

namespace my_custom_ops {
using namespace mindspore;
using namespace mindspore::kernel;
using namespace mindspore::device::ascend;
using namespace mindspore::ops;

class OPS_API InplaceAddOpFuncImpl : public OpFuncImpl {
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

class InplaceAddAscend : public AclnnKernelMod {
 public:
  InplaceAddAscend() : AclnnKernelMod(std::move("aclnnInplaceAdd")) {}

  ~InplaceAddAscend() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(stream_ptr);
    RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], this->alpha_);
    return true;
  }

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    auto alpha_dtype_id = inputs[kIndex2]->dtype_id();
    switch (alpha_dtype_id) {
      case kNumberTypeBool: {
        auto alpha_value = inputs[kIndex2]->GetValueWithCheck<bool>();
        MAKE_SCALAR(alpha_value, alpha_dtype_id, alpha_);
        break;
      }
      case kNumberTypeFloat32: {
        auto alpha_value = inputs[kIndex2]->GetValueWithCheck<float>();
        MAKE_SCALAR(alpha_value, alpha_dtype_id, alpha_);
        break;
      }
      case kNumberTypeFloat64: {
        auto alpha_value = inputs[kIndex2]->GetValueWithCheck<double>();
        MAKE_SCALAR(alpha_value, alpha_dtype_id, alpha_);
        break;
      }
      case kNumberTypeInt64: {
        auto alpha_value = inputs[kIndex2]->GetValueWithCheck<int64_t>();
        MAKE_SCALAR(alpha_value, alpha_dtype_id, alpha_);
        break;
      }
      default:
        MS_LOG(EXCEPTION) << "AddExt only support bool, float32, float64 and int64, but got " << alpha_dtype_id;
    }
    GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], alpha_);
  }

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE()

  ScalarPtr alpha_ = nullptr;
};
}  // namespace my_custom_ops

REG_GRAPH_MODE_OP(inplace_add, my_custom_ops::InplaceAddOpFuncImpl, my_custom_ops::InplaceAddAscend);
