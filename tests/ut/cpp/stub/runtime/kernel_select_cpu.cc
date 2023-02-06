#include <string>
#include "utils/log_adapter.h"
#include "ir/anf.h"

namespace mindspore {
namespace device {
namespace cpu {
bool IsDynamicParamKernel(const std::string &op_name) { return false; }

std::pair<std::string, ExceptionType> SetKernelInfoWithMsg(const CNodePtr &apply_kernel_ptr) { return {}; }
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
