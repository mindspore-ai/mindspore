#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_AUTO_GENERATE_PYBOOST_NATIVE_GRAD_FUNCTIONS
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_AUTO_GENERATE_PYBOOST_NATIVE_GRAD_FUNCTIONS

#include <map>
#include <string>
#include <vector>
#include "kernel/pyboost/op_runner.h"
#include "runtime/pynative/op_runner.h"
#include "frontend/expander/bprop/bprop_irbuilder.h"

namespace mindspore{
namespace pynative {
using NodePtr = expander::NodePtr;
using NodePtrList = std::vector<expander::NodePtr>;
class NativeFunc {
  public:
    static const std::string &device_target() { return device_target_;}
    static void set_device_target(const std::string &device_target) { device_target_ = device_target; }
    static NodePtr RunOpInVm(const PrimitivePtr &prim, const NodePtrList &inputs);
    static NodePtr RunOpDeprecated(const PrimitivePtr &prim, const NodePtrList &inputs);
    ${native_grad_func_def};
  private:
    static std::string device_target_;
};
}
}
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_AUTO_GENERATE_PYBOOST_NATIVE_GRAD_FUNCTIONS