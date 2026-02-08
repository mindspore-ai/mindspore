#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_AUTO_GENERATE_PYBOOST_NATIVE_GRAD_FUNCTIONS
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_AUTO_GENERATE_PYBOOST_NATIVE_GRAD_FUNCTIONS

#include <map>
#include <string>
#include <vector>
#include "include/pynative/utils/pyboost/op_runner.h"
#include "include/pynative/utils/runtime/op_runner.h"
#include "include/pynative/utils/pyboost/comm_handle.h"
#include "include/frontend/expander/bprop_interface.h"

namespace mindspore{
namespace pynative {
using NodePtr = expander::NodePtr;
using Emitter = expander::Emitter;
using NodePtrList = std::vector<expander::NodePtr>;

using CommHandle = kernel::pyboost::CommHandle;
class CommFuncNode : public expander::FuncNode {
 public:
  CommFuncNode(const ValuePtr &value, const abstract::AbstractBasePtr &abs, InputType input_type, Emitter *emitter, std::shared_ptr<CommHandle> comm_handle)
      : FuncNode(value, abs, input_type, emitter), comm_handle_(comm_handle) {}
  void set_comm_handle(const std::shared_ptr<CommHandle> &comm_handle) { comm_handle_ = comm_handle; }
  std::shared_ptr<CommHandle> comm_handle() { return comm_handle_; }
 private:
  std::shared_ptr<CommHandle> comm_handle_{nullptr};
};



class NativeFunc {
  public:
    static device::DeviceType device_target() { return device_target_;}
    static void set_device_target(device::DeviceType device_target) { device_target_ = device_target; }
    static NodePtr RunOpInVm(const PrimitivePtr &prim, const NodePtrList &inputs);
    static NodePtr RunOpDeprecated(const PrimitivePtr &prim, const NodePtrList &inputs);
    static ValuePtr ConvertNode2Value(const NodePtr &node);
    ${native_grad_func_def};
  private:
    static device::DeviceType device_target_;
};
}
}
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_AUTO_GENERATE_PYBOOST_NATIVE_GRAD_FUNCTIONS
