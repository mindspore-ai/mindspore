//
// Created by jojo on 24-6-7.
//

#include "runtime/pipeline/task/task.h"
#include <atomic>

namespace mindspore {
namespace runtime {
uint64_t AsyncTask::MakeId() {
  static std::atomic<uint64_t> last_id{1};
  return last_id.fetch_add(1, std::memory_order_relaxed);
}
}  // namespace runtime
}  // namespace mindspore