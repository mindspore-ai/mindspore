#include <string>

namespace mindspore {
namespace device {
namespace cpu {

bool IsDynamicParamKernel(const std::string &op_name) { return false; }

}  // namespace cpu
}  // namespace device
}  // namespace mindspore
