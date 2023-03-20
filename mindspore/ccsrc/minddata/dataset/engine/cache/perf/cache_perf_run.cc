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

#include "minddata/dataset/engine/cache/perf/cache_perf_run.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <sstream>
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/services.h"
#include "minddata/dataset/util/sig_handler.h"

constexpr int64_t field_width_ten = 10;
constexpr int64_t field_width_eleven = 11;
constexpr int64_t field_width_twelve = 12;
constexpr int64_t field_width_thirteen = 13;
constexpr int64_t field_width_fourteen = 14;
constexpr int64_t field_width_sixteen = 16;
constexpr int64_t field_width_eighteen = 18;

namespace mindspore {
namespace dataset {
const char CachePerfRun::kCachePipelineBinary[] = "cache_pipeline";
const int32_t port_opt = 1000;      // there is no short option for port
const int32_t hostname_opt = 1001;  // there is no short option for hostname
const int32_t connect_opt = 1002;   // there is no short option for connect

void CachePerfRun::PrintHelp() {
  std::cout << "Options:\n"
               "    -h,--help:           Show this usage message\n"
               "    -s,--num_rows:       Set the sample size, i.e., the number of "
               "rows\n"
               "    -r,--row_size:       Set the average row size\n"
               "    -n,--pipeline:       Set the number of parallel pieplines. Default = "
            << kDftNumOfPipelines
            << "\n"
               "    -e,--epoch:          Set the number of epochs. Default = "
            << kDftNumberOfEpochs
            << "\n"
               "       --shuffle:        Set shuffle=True. Default = "
            << std::boolalpha << kDftShuffle
            << "\n"
               "    -p,--prefetch_size:  Set the prefetch size for cache. Default = "
            << kDftCachePrefetchSize << "\n"
            << "    -a,--cache_size:     Set cache size. Default = " << kDftCacheSize
            << " (Mb)\n"
               "       --spill:          Set spill to disk to True. Default = "
            << std::boolalpha << kDftSpill << "\n"
            << "    -w,--workers:        Set the number of parallel workers. Default = " << cfg_.num_parallel_workers()
            << "\n"
               "       --connection:     Set number of TCP/IP connections per pipeline. Default = "
            << kDftNumConnections << "\n"
            << "       --port:           TCP/IP port of the cache server. Default = " << kCfgDefaultCachePort << "\n"
            << "       --hostname:       Hostname of the cache server. Default = " << kCfgDefaultCacheHost << "\n";
}

int32_t CachePerfRun::ProcessArgsHelper(int32_t opt) {
  int32_t rc = 0;
  try {
    switch (opt) {
      case 'n': {
        num_pipelines_ = std::stoi(optarg);
        break;
      }

      case 'e': {
        num_epoches_ = std::stoi(optarg);
        break;
      }

      case 'p': {
        int32_t prefetch_sz = std::stoi(optarg);
        cache_builder_.SetPrefetchSize(prefetch_sz);
        break;
      }

      case 'a': {
        int32_t cache_sz = std::stoi(optarg);
        cache_builder_.SetCacheMemSz(cache_sz);
        break;
      }

      case 's': {
        num_rows_ = std::stoi(optarg);
        break;
      }

      case 'r': {
        row_size_ = std::stoi(optarg);
        break;
      }

      case 'w': {
        cfg_.set_num_parallel_workers(std::stoi(optarg));
        break;
      }

      case connect_opt: {
        int32_t connection_sz = std::stoi(optarg);
        cache_builder_.SetNumConnections(connection_sz);
        break;
      }

      case port_opt: {
        int32_t port = std::stoi(optarg);
        cache_builder_.SetPort(port);
        break;
      }

      case hostname_opt: {
        std::string hostname = optarg;
        cache_builder_.SetHostname(hostname);
        break;
      }

      case 'h':  // -h or --help
        PrintHelp();
        rc = -1;
        break;

      case ':':
        std::cerr << "Missing argument for option " << char(optopt) << std::endl;
        rc = -1;
        break;

      case '?':  // Unrecognized option
      default:
        std::cerr << "Unknown option " << char(optopt) << std::endl;
        PrintHelp();
        rc = -1;
        break;
    }
  } catch (const std::exception &e) {
    PrintHelp();
    rc = -1;
  }
  return rc;
}

int32_t CachePerfRun::SanityCheck(std::map<int32_t, int32_t> seen_opts) {
  // We have all the defaults except sample size and average row size which the user must specify.
  auto it = seen_opts.find('s');
  if (it == seen_opts.end()) {
    std::cerr << "Missing sample size." << std::endl;
    return -1;
  }

  it = seen_opts.find('r');
  if (it == seen_opts.end()) {
    std::cerr << "Missing average row size." << std::endl;
    return -1;
  }

  if (num_rows_ <= 0) {
    std::cerr << "Sample size must be positive." << std::endl;
    return -1;
  }

  if (row_size_ <= 0) {
    std::cerr << "Average row size must be positive." << std::endl;
    return -1;
  }

  if (num_pipelines_ <= 0) {
    std::cerr << "Number of pipelines must be positive." << std::endl;
    return -1;
  }

  if (num_epoches_ <= 0) {
    std::cerr << "Number of epoches must be positive." << std::endl;
    return -1;
  }

  if (num_rows_ < num_pipelines_) {
    std::cerr << "Sample size is smaller than the number of pipelines." << std::endl;
    return -1;
  }
  return 0;
}

int32_t CachePerfRun::ProcessArgs(int argc, char **argv) {
  if (argc == 1) {
    PrintHelp();
    return -1;
  }

  int shuffle = 0;
  int spill = 0;

  const char *const short_opts = ":n:e:p:a:s:r:w:";
  const option long_opts[] = {{"pipeline", required_argument, nullptr, 'n'},
                              {"epoch", required_argument, nullptr, 'e'},
                              {"prefetch_size", required_argument, nullptr, 'p'},
                              {"shuffle", no_argument, &shuffle, 1},
                              {"cache_size", required_argument, nullptr, 'a'},
                              {"num_rows", required_argument, nullptr, 's'},
                              {"row_size", required_argument, nullptr, 'r'},
                              {"workers", required_argument, nullptr, 'w'},
                              {"port", required_argument, nullptr, port_opt},
                              {"hostname", required_argument, nullptr, hostname_opt},
                              {"spill", no_argument, &spill, 1},
                              {"connection", required_argument, nullptr, connect_opt},
                              {"help", no_argument, nullptr, 'h'},
                              {nullptr, no_argument, nullptr, 0}};

  std::map<int32_t, int32_t> seen_opts;
  int32_t rc = 0;
  try {
    while (rc == 0) {
      int32_t option_indxex;
      const auto opt = getopt_long(argc, argv, short_opts, long_opts, &option_indxex);

      if (opt == -1) {
        if (optind < argc) {
          rc = -1;
          std::cerr << "Unknown arguments: ";
          while (optind < argc) {
            std::cerr << argv[optind++] << " ";
          }
          std::cerr << std::endl;
        }
        break;
      }

      if (opt > 0) {
        seen_opts[opt]++;
        if (seen_opts[opt] > 1) {
          std::string long_name = long_opts[option_indxex].name;
          std::cerr << "The " << long_name << " argument was given more than once." << std::endl;
          rc = -1;
          continue;
        }
      }

      if (opt == 0) {
        if (long_opts[option_indxex].flag == &shuffle) {
          shuffle_ = true;
        } else if (long_opts[option_indxex].flag == &spill) {
          cache_builder_.SetSpill(true);
        }
        continue;
      }

      rc = ProcessArgsHelper(opt);
    }
  } catch (const std::exception &e) {
    PrintHelp();
    rc = -1;
  }
  if (rc < 0) {
    return rc;
  }

  rc = SanityCheck(seen_opts);
  if (rc < 0) {
    return rc;
  }

  pid_lists_.reserve(num_pipelines_);
  return 0;
}

Status CachePerfRun::GetSession() {
  CacheClientGreeter comm(cache_builder_.GetHostname(), cache_builder_.GetPort(), 1);
  RETURN_IF_NOT_OK(comm.ServiceStart());
  auto rq = std::make_shared<GenerateSessionIdRequest>();
  RETURN_IF_NOT_OK(comm.HandleRequest(rq));
  RETURN_IF_NOT_OK(rq->Wait());
  session_ = rq->GetSessionId();
  std::cout << "Session: " << session_ << std::endl;
  cache_builder_.SetSessionId(session_);
  return Status::OK();
}

CachePerfRun::CachePerfRun()
    : my_pipeline_(-1),
      num_pipelines_(kDftNumOfPipelines),
      num_epoches_(kDftNumberOfEpochs),
      num_rows_(0),
      row_size_(0),
      shuffle_(kDftShuffle),
      session_(0),
      crc_(0),
      epoch_sync_cnt_(0) {
  cache_builder_.SetSpill(kDftSpill).SetCacheMemSz(kDftCacheSize);
}

CachePerfRun::~CachePerfRun() {
  if (session_ != 0) {
    Status rc;
    CacheClientGreeter comm(cache_builder_.GetHostname(), cache_builder_.GetPort(), 1);
    rc = comm.ServiceStart();
    if (rc.IsOk()) {
      CacheClientInfo cinfo;
      cinfo.set_session_id(session_);
      auto rq = std::make_shared<DropSessionRequest>(cinfo);
      rc = comm.HandleRequest(rq);
      if (rc.IsOk()) {
        rc = rq->Wait();
        if (rc.IsOk()) {
          std::cout << "Drop session " << session_ << " successful" << std::endl;
        }
      }
    }
  }
  // Send an interrupt message to each child.
  for (auto msg_qid : msg_send_lists_) {
    CachePerfMsg msg;
    msg.SetType(CachePerfMsg::MessageType::kInterrupt);
    (void)msg.Send(msg_qid);
  }
  // Wait for each child to return
  for (auto pid : pid_lists_) {
    int status;
    if (waitpid(pid, &status, 0) == -1) {
      std::string errMsg = "waitpid fails. errno = " + std::to_string(errno);
      std::cerr << errMsg << std::endl;
    } else {
      MS_LOG(INFO) << "Child pid " << pid << " returns." << std::endl;
    }
  }
  // Remove all the message queues
  for (auto msg_qid : msg_send_lists_) {
    // Remove the message que and never mind about the return code.
    (void)msgctl(msg_qid, IPC_RMID, nullptr);
  }
  for (auto msg_qid : msg_recv_lists_) {
    // Remove the message que and never mind about the return code.
    (void)msgctl(msg_qid, IPC_RMID, nullptr);
  }
}

void CachePerfRun::PrintEpochSummary() const {
  std::cout << std::setw(field_width_twelve) << "Pipeline #" << std::setw(field_width_ten) << "worker id"
            << std::setw(field_width_eleven) << "min (μs)" << std::setw(field_width_eleven) << "max (μs)"
            << std::setw(field_width_eleven) << "avg (μs)" << std::setw(field_width_fourteen) << "median (μs)"
            << std::setw(field_width_fourteen) << "buffer count" << std::setw(field_width_eighteen)
            << "Elapsed time (s)" << std::endl;
  for (auto &it : epoch_results_) {
    auto epoch_worker_summary = it.second;
    std::cout << std::setw(field_width_twelve) << (epoch_worker_summary.pipeline() + 1) << std::setw(field_width_ten)
              << epoch_worker_summary.worker() << std::setw(field_width_ten) << epoch_worker_summary.min()
              << std::setw(field_width_ten) << epoch_worker_summary.max() << std::setw(field_width_ten)
              << epoch_worker_summary.avg() << std::setw(field_width_thirteen) << epoch_worker_summary.med()
              << std::setw(field_width_fourteen) << epoch_worker_summary.cnt() << std::setw(field_width_eighteen)
              << epoch_worker_summary.elapse() << std::endl;
  }
}

Status CachePerfRun::ListenToPipeline(int32_t workerId) {
  TaskManager::FindMe()->Post();
  int32_t qID = msg_recv_lists_[workerId];
  do {
    RETURN_IF_INTERRUPTED();
    CachePerfMsg msg;
    RETURN_IF_NOT_OK(msg.Receive(qID));
    // Decode the messages.
    auto type = msg.GetType();
    char *p = msg.GetMutableBuffer();
    switch (type) {
      case CachePerfMsg::MessageType::kEpochResult: {
        PipelineWorkerEpochSummary epoch_worker_summary;
        CHECK_FAIL_RETURN_UNEXPECTED(epoch_worker_summary.ParseFromArray(p, msg.GetProtoBufSz()), "Parse fail");
        {
          auto pipeline = epoch_worker_summary.pipeline();
          auto worker = epoch_worker_summary.worker();
          std::unique_lock<std::mutex> lock(mux_);
          // sort by pipeline/worker
          auto r =
            epoch_results_.emplace(std::pair<int32_t, int32_t>(pipeline, worker), std::move(epoch_worker_summary));
          CHECK_FAIL_RETURN_UNEXPECTED(r.second, "Insert failed");
        }
        break;
      }
      case CachePerfMsg::MessageType::kEpochEnd: {
        EpochDone proto;
        CHECK_FAIL_RETURN_UNEXPECTED(proto.ParseFromArray(p, msg.GetProtoBufSz()), "Parse fail");
        auto n = epoch_sync_cnt_.fetch_add(1);
        if (n + 1 == num_pipelines_) {
          pipeline_wp_.Set();
        }
        break;
      }
      case CachePerfMsg::MessageType::kInterrupt: {
        TaskManager::WakeUpWatchDog();
        return Status::OK();
      }
      case CachePerfMsg::kError: {
        ErrorMsg proto;
        CHECK_FAIL_RETURN_UNEXPECTED(proto.ParseFromArray(p, msg.GetProtoBufSz()), "Parse fail");
        return Status(static_cast<const StatusCode>(proto.rc()), proto.msg());
      }
      default:
        std::string errMsg = "Unknown request type: " + std::to_string(type);
        MS_LOG(ERROR) << errMsg;
        RETURN_STATUS_UNEXPECTED(errMsg);
        break;
    }
  } while (true);
  return Status::OK();
}

Status CachePerfRun::StartPipelines() {
  for (auto i = 0; i < num_pipelines_; ++i) {
    auto pid = fork();
    if (pid == 0) {
      // Child. We will call another binary but with different (hidden) parameters.
      // The parent process is waiting on a wait post. Any error we hit here we must interrupt the
      // parent process
      auto interrupt_parent = [this, i]() {
        CachePerfMsg msg;
        msg.SetType(CachePerfMsg::MessageType::kInterrupt);
        msg.Send(msg_recv_lists_[i]);
      };
      const std::string self_proc = "/proc/self/exe";
      std::string canonical_path;
      canonical_path.resize(400);  // PATH_MAX is large. This value should be big enough for our use.
      // Some lower level OS library calls are needed here to determine the binary path.
      if (realpath(self_proc.data(), canonical_path.data()) == nullptr) {
        std::cerr << "Failed to identify cache_perf binary path: " + std::to_string(errno) << ": " << strerror(errno)
                  << std::endl;
        interrupt_parent();
        // Call _exit instead of exit because we will hang in TaskManager destructor for a forked child process.
        _exit(-1);
      }
      canonical_path.resize(strlen(canonical_path.data()));
      int last_seperator = canonical_path.find_last_of('/');
      if (last_seperator == std::string::npos) {
        std::cerr << "Canonical path can't locate / " << canonical_path << std::endl;
        interrupt_parent();
        // Call _exit instead of exit because we will hang in TaskManager destructor for a forked child process.
        _exit(-1);
      }
      // truncate the binary name so we are left with the absolute path of cache_admin binary
      canonical_path.resize(last_seperator + 1);
      std::string cache_pipeline_binary = canonical_path + std::string(kCachePipelineBinary);

      std::string pipeline_cfg = std::to_string(i) + "," + std::to_string(session_) + "," + std::to_string(crc_) + "," +
                                 std::to_string(msg_send_lists_[i]) + "," + std::to_string(msg_recv_lists_[i]) + "," +
                                 std::to_string(num_pipelines_) + "," + std::to_string(num_epoches_) + "," +
                                 std::to_string(num_rows_) + "," + std::to_string(row_size_) + "," +
                                 std::to_string(cfg_.num_parallel_workers()) + "," +
                                 (shuffle_ ? std::string("true").data() : std::string("false").data());
      std::string client_cfg = cache_builder_.GetHostname() + "," + std::to_string(cache_builder_.GetPort()) + "," +
                               std::to_string(cache_builder_.GetPrefetchSize()) + "," +
                               std::to_string(cache_builder_.GetCacheMemSz()) + "," +
                               std::to_string(cache_builder_.GetNumConnections()) + "," +
                               (cache_builder_.isSpill() ? std::string("true").data() : std::string("false").data());
      char *argv[4];
      argv[0] = const_cast<char *>(kCachePipelineBinary);
      argv[1] = pipeline_cfg.data();
      argv[2] = client_cfg.data();
      argv[3] = nullptr;
      // Invoke the binary.
      execv(cache_pipeline_binary.data(), argv);
      std::cerr << "Unable to exec. Errno = " + std::to_string(errno) << ": " << strerror(errno) << std::endl;
      interrupt_parent();
      // Call _exit instead of exit because we will hang TaskManager destructor for a forked child process.
      _exit(-1);
    } else if (pid > 0) {
      std::cout << "Pipeline number " << (i + 1) << " has been created with process id: " << pid << std::endl;
      pid_lists_.push_back(pid);
    } else {
      std::string errMsg = "Failed to fork process for cache pipeline: " + std::to_string(errno);
      RETURN_STATUS_UNEXPECTED(errMsg);
    }
  }
  return Status::OK();
}

Status CachePerfRun::Cleanup() {
  // Destroy the cache. We no longer need it around.
  RETURN_IF_NOT_OK(cc_->DestroyCache());

  // Unreserve the session
  CacheClientInfo cinfo;
  cinfo.set_session_id(session_);
  auto rq = std::make_shared<DropSessionRequest>(cinfo);
  RETURN_IF_NOT_OK(cc_->PushRequest(rq));
  RETURN_IF_NOT_OK(rq->Wait());
  std::cout << "Drop session " << session_ << " successful" << std::endl;
  session_ = 0;
  return Status::OK();
}

Status CachePerfRun::Run() {
  // Now we bring up TaskManager.
  RETURN_IF_NOT_OK(Services::CreateInstance());
  // Handle Control-C
  RegisterHandlers();

  // Get a session from the server.
  RETURN_IF_NOT_OK(GetSession());

  // Generate a random crc.
  auto mt = GetRandomDevice();
  std::uniform_int_distribution<session_id_type> distribution(0, std::numeric_limits<int32_t>::max());
  crc_ = distribution(mt);
  std::cout << "CRC: " << crc_ << std::endl;

  // Create all the resources required by the pipelines before we fork.
  for (auto i = 0; i < num_pipelines_; ++i) {
    // We will use shared message queues for communication between parent (this process)
    // and each pipelines.
    auto access_mode = S_IRUSR | S_IWUSR;
    int32_t msg_send_qid = msgget(IPC_PRIVATE, IPC_CREAT | IPC_EXCL | access_mode);
    if (msg_send_qid == -1) {
      std::string errMsg = "Unable to create a message queue. Errno = " + std::to_string(errno);
      RETURN_STATUS_UNEXPECTED(errMsg);
    }
    msg_send_lists_.push_back(msg_send_qid);
    int32_t msg_recv_qid = msgget(IPC_PRIVATE, IPC_CREAT | IPC_EXCL | access_mode);
    if (msg_recv_qid == -1) {
      std::string errMsg = "Unable to create a message queue. Errno = " + std::to_string(errno);
      RETURN_STATUS_UNEXPECTED(errMsg);
    }
    msg_recv_lists_.push_back(msg_recv_qid);
  }

  // Now we create the children knowing all two sets of message queues are constructed.
  auto start_tick = std::chrono::steady_clock::now();
  RETURN_IF_NOT_OK(StartPipelines());

  // Spawn a few threads to monitor the communications from the pipeline.
  RETURN_IF_NOT_OK(vg_.ServiceStart());
  auto f = std::bind(&CachePerfRun::ListenToPipeline, this, std::placeholders::_1);
  for (auto i = 0; i < num_pipelines_; ++i) {
    RETURN_IF_NOT_OK(vg_.CreateAsyncTask("Queue listener", std::bind(f, i)));
  }

  // Wait until all pipelines finish the first epoch.
  RETURN_IF_NOT_OK(pipeline_wp_.Wait());
  auto end_tick = std::chrono::steady_clock::now();

  int64_t elapse_time = std::chrono::duration_cast<std::chrono::seconds>(end_tick - start_tick).count();
  std::cout << "Epoch one (build phase) elapsed time " << elapse_time << " seconds" << std::endl;

  std::cout << "Epoch one (build phase) per pipeline per worker summary." << std::endl;
  PrintEpochSummary();

  // Get some stat but we need to connect. The server will thinks it is the (n+1) pipeline
  RETURN_IF_NOT_OK(cache_builder_.Build(&cc_));
  Status rc = cc_->CreateCache(crc_, false);
  // Duplicate key is fine.
  if (rc.IsError() && rc != StatusCode::kMDDuplicateKey) {
    return rc;
  }

  CacheServiceStat stat{};
  RETURN_IF_NOT_OK(cc_->GetStat(&stat));

  std::cout << "Get statistics for this session:\n";
  std::cout << std::setw(field_width_twelve) << "Mem cached" << std::setw(field_width_twelve) << "Disk cached"
            << std::setw(field_width_sixteen) << "Avg cache size" << std::setw(field_width_ten) << "Numa hit"
            << std::endl;
  std::string stat_mem_cached = (stat.num_mem_cached == 0) ? "n/a" : std::to_string(stat.num_mem_cached);
  std::string stat_disk_cached = (stat.num_disk_cached == 0) ? "n/a" : std::to_string(stat.num_disk_cached);
  std::string stat_avg_cached = (stat.avg_cache_sz == 0) ? "n/a" : std::to_string(stat.avg_cache_sz);
  std::string stat_numa_hit = (stat.num_numa_hit == 0) ? "n/a" : std::to_string(stat.num_numa_hit);

  std::cout << std::setw(12) << stat_mem_cached << std::setw(12) << stat_disk_cached << std::setw(16) << stat_avg_cached
            << std::setw(10) << stat_numa_hit << std::endl;

  // Toggle write mode off since the rest are just read only.
  // Simplest way is call this special internal function.
  cc_->ServerRunningOutOfResources();

  // The rest of the epochs are just fetching.
  auto epoch_num = 2;
  while (epoch_num <= num_epoches_) {
    epoch_sync_cnt_ = 0;
    pipeline_wp_.Clear();
    epoch_results_.clear();
    start_tick = std::chrono::steady_clock::now();
    // Signal each pipeline to start
    for (auto msg_qid : msg_send_lists_) {
      CachePerfMsg msg;
      msg.SetType(CachePerfMsg::MessageType::kEpochStart);
      (void)msg.Send(msg_qid);
    }
    // Wait for the child to finish
    RETURN_IF_NOT_OK(pipeline_wp_.Wait());
    end_tick = std::chrono::steady_clock::now();
    elapse_time = std::chrono::duration_cast<std::chrono::seconds>(end_tick - start_tick).count();
    std::cout << "Epoch " << epoch_num << " elapsed time " << elapse_time << " seconds" << std::endl;
    std::cout << "Epoch " << epoch_num
              << " (read phase) per pipeline per worker summary. Buffer size = " << cc_->GetPrefetchSize() << std::endl;
    PrintEpochSummary();
    ++epoch_num;
  }

  // Destroy the cache client and drop the session
  RETURN_IF_NOT_OK(Cleanup());
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
