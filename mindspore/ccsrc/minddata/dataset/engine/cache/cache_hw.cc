/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#include "minddata/dataset/engine/cache/cache_hw.h"
#ifdef NUMA_ENABLED
#include <numa.h>
#endif
#include <sched.h>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <fstream>
#include <regex>
#include <thread>
#include "utils/log_adapter.h"
namespace mindspore {
namespace dataset {
CacheServerHW::CacheServerHW() {
  num_cpus_ = std::thread::hardware_concurrency();
  MS_LOG(DEBUG) << "Number of cpu(s) : " << num_cpus_;
#ifdef NUMA_ENABLED
  if (numa_enabled()) {
    MS_LOG(INFO) << "Numa support enabled";
    for (auto i = 0; i <= numa_max_node(); ++i) {
      int64_t free_avail;
      int64_t mem_avail = numa_node_size(i, &free_avail);
      MS_LOG(INFO) << "Total physical/free RAM in bytes at node " << i << " : " << mem_avail << "/" << free_avail;
    }
  }
#endif
}

int64_t CacheServerHW::GetTotalSystemMemory() {
  auto pages = sysconf(_SC_PHYS_PAGES);
  auto page_size = sysconf(_SC_PAGE_SIZE);
  auto total = static_cast<int64_t>(pages) * static_cast<int64_t>(page_size);
  MS_LOG(INFO) << "Total physical RAM in bytes: " << total;
  return total;
}

Status CacheServerHW::SetDefaultMemoryPolicy(CachePoolPolicy policy) {
#ifdef NUMA_ENABLED
  if (numa_enabled()) {
    // Set our default memory policy.
    switch (policy) {
      case kLocal:
        numa_set_localalloc();
        MS_LOG(DEBUG) << "Setting memory default policy to local node. Low level code may override the setting";
        break;
      case kInterleave:
        numa_set_interleave_mask(numa_all_nodes_ptr);
        MS_LOG(DEBUG) << "Numa affinity is turned off. Use interleave memory policy as default.";
        break;
      case kOnNode:
      case kPreferred:
        RETURN_STATUS_UNEXPECTED("Unsupported memory policy");
        break;
      case kNone:
      default:
        // No action taken.
        break;
    }
  }
#endif
  return Status::OK();
}

Status CacheServerHW::GetNumaNodeInfo() {
  std::set<Path> numa_nodes_;
  Path node(kSysNodePath);
  auto it = Path::DirIterator::OpenDirectory(&node);
  if (it == nullptr) {
    MS_LOG(WARNING) << "Unable to open directory " << kSysNodePath << ". Skip scanning hardware info";
    return Status::OK();
  }
  auto isdigit_string = [](const char *str) -> bool {
    bool r = true;
    for (auto i = 0; i < strlen(str); ++i) {
      if (!std::isdigit(str[i])) {
        r = false;
        break;
      }
    }
    return r;
  };
  // Look for name starts with 'node' and followed by digits.
  const char kNodeName[] = "node";
  while (it->hasNext()) {
    auto p = it->next();
    const std::string entry = p.Basename();
    const char *name = entry.data();
    if (strncmp(name, kNodeName, 4) == 0 && isdigit_string(name + strlen(kNodeName))) {
      numa_nodes_.insert(p);
    }
  }
  // There should be at least one. But if not found in any case, just move on the
  // rest of the server start up.
  if (numa_nodes_.empty()) {
    MS_LOG(WARNING) << "No numa nodes ? Skip scanning hardware info";
    return Status::OK();
  }
  // For each numa node, get a list of CPU that is associated with it.
  const char kCpuList[] = "cpulist";
  auto r = std::regex("[0-9]*-[0-9]*");
  for (Path p : numa_nodes_) {
    auto node_dir = p.Basename();
    numa_id_t numa_node = strtol(node_dir.data() + strlen(kNodeName), nullptr, 10);
    Path f = p / kCpuList;
    std::ifstream fs(f.toString());
    CHECK_FAIL_RETURN_UNEXPECTED(!fs.fail(), "Fail to open file: " + f.toString());
    std::string cpu_string;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    int32_t cpu_cnt = 0;
    while (getline(fs, cpu_string)) {
      // Now we parse the content of cpu_string.
      std::sregex_iterator iter(cpu_string.begin(), cpu_string.end(), r);
      std::sregex_iterator end;
      while (iter != end) {
        auto match = iter->str();
        auto pos = match.find_first_of('-');
        CHECK_FAIL_RETURN_UNEXPECTED(pos != std::string::npos, "Failed to parse numa node file");
        std::string min = match.substr(0, pos);
        std::string max = match.substr(pos + 1);
        cpu_id_t cpu_min = strtol(min.data(), nullptr, 10);
        cpu_id_t cpu_max = strtol(max.data(), nullptr, 10);
        MS_LOG(DEBUG) << "Numa node " << numa_node << " CPU(s) : " << cpu_min << "-" << cpu_max;
        for (int i = cpu_min; i <= cpu_max; ++i) {
          CPU_SET(i, &cpuset);
          ++cpu_cnt;
        }
        ++iter;
      }
    }
    CHECK_FAIL_RETURN_UNEXPECTED(!fs.bad(), "Fail to read file: " + f.toString());
    fs.close();
    // Remember which cpu is attached to this numa node.
    numa_cpuset_.emplace(numa_node, cpuset);
    numa_cpu_cnt_.emplace(numa_node, cpu_cnt);
  }
  MS_LOG(DEBUG) << "Number of numa nodes : " << numa_cpuset_.size();
  return Status::OK();
}

Status CacheServerHW::SetAffinity(const Task &tk, numa_id_t numa_node) {
#if defined(__APPLE__)
  return Status::OK();
#else
  auto r = numa_cpuset_.find(numa_node);
  if (r != numa_cpuset_.end()) {
    auto err = pthread_setaffinity_np(tk.GetNativeHandle(), sizeof(r->second), &r->second);
    if (err) {
      std::string errMsg = "Unable to set affiity. Errno = " + std::to_string(errno);
      RETURN_STATUS_UNEXPECTED(errMsg);
    }
  } else {
    RETURN_STATUS_UNEXPECTED("Numa node " + std::to_string(numa_node) + " not found");
  }
  return Status::OK();
#endif
}

std::vector<cpu_id_t> CacheServerHW::GetCpuList(numa_id_t numa_id) {
  std::vector<cpu_id_t> v;
  auto it = numa_cpuset_.find(numa_id);
  if (it != numa_cpuset_.end()) {
    auto &cpu_set = it->second;
    for (auto i = 0; i < num_cpus_; ++i) {
      if (CPU_ISSET(i, &cpu_set)) {
        v.push_back(i);
      }
    }
  }
  return v;
}

numa_id_t CacheServerHW::GetMyNode() const {
#if defined(__APPLE__)
  numa_id_t node_id = -1;
#else
  numa_id_t node_id = 0;
  auto cpu = sched_getcpu();
#ifdef NUMA_ENABLED
  node_id = numa_node_of_cpu(cpu);
#else
  bool found = false;
  for (auto it : numa_cpuset_) {
    cpu_set_t &cpu_set = it.second;
    if (CPU_ISSET(cpu, &cpu_set)) {
      node_id = it.first;
      found = true;
      break;
    }
  }
  MS_LOG(DEBUG) << "cpu id " << cpu << " found : " << std::boolalpha << found;
#endif  // end NUMA_ENABLED
#endif  // end __APPLE__
  return node_id;
}

void CacheServerHW::InterleaveMemory(void *ptr, size_t sz) {
#ifdef NUMA_ENABLED
  if (numa_enabled()) {
    numa_interleave_memory(ptr, sz, numa_all_nodes_ptr);
  }
#endif
}

void CacheServerHW::AssignToNode(numa_id_t numa_id, void *ptr, size_t sz) {
#ifdef NUMA_ENABLED
  if (numa_enabled()) {
    numa_tonode_memory(ptr, sz, numa_id);
  }
#endif
}

bool CacheServerHW::numa_enabled() {
#ifdef NUMA_ENABLED
  return (numa_available() != -1);
#else
  return false;
#endif
}
}  // namespace dataset
}  // namespace mindspore
