import mindspore.context as context
import mindspore as ms
import torch_npu
import psutil
import time


class TestStreamApi():
    "TestStreamApi"
    def bind_to_cpu(self):
        p = psutil.Process()
        p.cpu_affinity([0])

    def setup_method(self):
        context.set_context(mode=context.PYNATIVE_MODE)

        # Initialize streams and events for some APIS because MindSpore's constructor costs more.
        self.ms_wait_stream_stream = ms.runtime.Stream()
        self.pta_wait_stream_stream = torch_npu.npu.Stream()

        self.ms_stream_wait_event_event = ms.runtime.Event()
        self.ms_stream_wait_event_stream = ms.runtime.Stream()

        self.pta_stream_wait_event_event = torch_npu.npu.Event()
        self.pta_stream_wait_event_stream = torch_npu.npu.Stream()

    def framework(self, test_name, ms_func, torch_func, count=1000):
        start_max_device_mem = ms.runtime.max_memory_allocated()
        ms_func()
        end_max_device_mem = ms.runtime.max_memory_allocated()
        ms_cost_device_mem = end_max_device_mem - start_max_device_mem

        ms.runtime.synchronize()

        # profiler = Profiler(output_path="./" + test_name, profile_framework="all")
        start_time = time.time()
        for _ in range(count):
            ms_func()
        ms.runtime.synchronize()
        end_time = time.time()
        ms_cost_time = end_time - start_time
        # profiler.analyse()

        start_max_device_mem = torch_npu.npu.max_memory_allocated()
        torch_func()
        end_max_device_mem = torch_npu.npu.max_memory_allocated()
        torch_cost_device_mem = end_max_device_mem - start_max_device_mem

        torch_npu.npu.synchronize()
        start_time = time.time()
        for _ in range(count):
            torch_func()
        torch_npu.npu.synchronize()
        end_time = time.time()
        torch_cost_time = end_time - start_time

        print("Test [{}] result:".format(test_name))
        print("\t[ MS ]  max device memory: {}, cost time:{}".format(ms_cost_device_mem, ms_cost_time), flush=True)
        print("\t[Torch] max device memory: {}, cost time:{}".format(torch_cost_device_mem, torch_cost_time),
              flush=True)

        # assert (ms_cost_time * 0.9) < torch_cost_time
        # assert ms_cost_device_mem <= torch_cost_device_mem

    def ms_current_stream(self):
        ms.runtime.current_stream()

    def torch_current_stream(self):
        torch_npu.npu.current_stream()

    def test_current_stream(self):
        self.framework("current_stream", self.ms_current_stream, self.torch_current_stream)

    def ms_default_stream(self):
        ms.runtime.default_stream()

    def torch_default_stream(self):
        torch_npu.npu.default_stream()

    def test_default_stream(self):
        self.framework("default_stream", self.ms_default_stream, self.torch_default_stream)

    def test_comm_stream(self):
        start_max_device_mem = ms.runtime.max_memory_allocated()
        _ = ms.runtime.communication_stream()
        end_max_device_mem = ms.runtime.max_memory_allocated()
        ms_cost_device_mem = end_max_device_mem - start_max_device_mem

        # profiler = Profiler(output_path="./" + test_name, profile_framework="all")
        start_time = time.time()
        for _ in range(1000):
            ms.runtime.communication_stream()
        end_time = time.time()
        ms_cost_time = end_time - start_time
        print("Test [communication stream ] result:")
        print("[MS] max device memory: {}, cost time:{}".format(ms_cost_device_mem, ms_cost_time), flush=True)

    def ms_sync(self):
        ms.runtime.synchronize()

    def torch_sync(self):
        torch_npu.npu.synchronize()

    def test_sync(self):
        self.framework("synchronize", self.ms_sync, self.torch_sync)

    def ms_stream(self):
        _ = ms.runtime.Stream()

    def torch_stream(self):
        _ = torch_npu.npu.Stream()

    def test_stream(self):
        self.framework("Stream", self.ms_stream, self.torch_stream)

    def ms_stream_sync(self):
        s1 = ms.runtime.current_stream()
        s1.synchronize()

    def torch_stream_sync(self):
        s1 = torch_npu.npu.current_stream()
        s1.synchronize()

    def test_stream_sync(self):
        self.framework("stream.sync", self.ms_stream_sync, self.torch_stream_sync)

    def ms_set_stream(self):
        s1 = ms.runtime.current_stream()
        ms.runtime.set_cur_stream(s1)

    def torch_set_stream(self):
        s1 = torch_npu.npu.current_stream()
        torch_npu.npu.set_stream(s1)

    def test_set_stream(self):
        self.framework("set_stream", self.ms_set_stream, self.torch_set_stream)

    def ms_stream_query(self):
        s1 = ms.runtime.current_stream()
        s1.query()

    def torch_stream_query(self):
        s1 = torch_npu.npu.current_stream()
        s1.query()

    def test_stream_query(self):
        self.framework("stream_query", self.ms_stream_query, self.torch_stream_query)

    def ms_wait_stream(self):
        s1 = ms.runtime.current_stream()
        s1.wait_stream(self.ms_wait_stream_stream)

    def torch_wait_stream(self):
        s1 = torch_npu.npu.current_stream()
        s1.wait_stream(self.pta_wait_stream_stream)

    def test_wait_stream(self):
        self.framework("wait_stream", self.ms_wait_stream, self.torch_wait_stream, 100)

    def ms_stream_ctx(self):
        s1 = ms.runtime.current_stream()
        with ms.runtime.StreamCtx(s1):
            pass

    def torch_stream_ctx(self):
        s1 = torch_npu.npu.current_stream()
        with torch_npu.npu.stream(s1):
            pass

    def test_stream_ctx(self):
        self.framework("stream ctx", self.ms_stream_ctx, self.torch_stream_ctx)

    def ms_stream_record_event(self):
        s1 = ms.runtime.current_stream()
        s1.record_event()

    def torch_stream_record_event(self):
        s1 = torch_npu.npu.current_stream()
        s1.record_event()

    def test_stream_record_event(self):
        self.framework("stream_record_event", self.ms_stream_record_event, self.torch_stream_record_event)

    def ms_stream_wait_event(self):
        cur_stream = ms.runtime.current_stream()
        ev = self.ms_stream_wait_event_event
        ev.record(cur_stream)

        s1 = self.ms_stream_wait_event_stream
        with ms.runtime.StreamCtx(s1):
            s1.wait_event(ev)

    def torch_stream_wait_event(self):
        cur_stream = torch_npu.npu.current_stream()
        ev = self.pta_stream_wait_event_event
        ev.record(cur_stream)

        s1 = self.pta_stream_wait_event_stream
        with torch_npu.npu.stream(s1):
            s1.wait_event(ev)

    def test_stream_wait_event(self):
        self.framework("stream_wait_event", self.ms_stream_wait_event, self.torch_stream_wait_event, 100)

    def ms_event(self):
        _ = ms.runtime.Event()
        _ = ms.runtime.Event(enable_timing=True, blocking=True)

    def torch_event(self):
        _ = torch_npu.npu.Event()
        _ = torch_npu.npu.Event(enable_timing=True, blocking=True)

    def test_event(self):
        self.framework("Event", self.ms_event, self.torch_event)

    def ms_event_record(self):
        ev1 = ms.runtime.Event()
        ev1.record()

    def torch_event_record(self):
        ev1 = torch_npu.npu.Event()
        ev1.record()

    def test_event_record(self):
        self.framework("event record", self.ms_event_record, self.torch_event_record)

    def ms_event_sync(self):

        ev1 = ms.runtime.Event()
        ev2 = ms.runtime.Event(enable_timing=True, blocking=True)
        ev1.record()
        ev2.record()
        ev1.synchronize()
        ev2.synchronize()

    def torch_event_sync(self):
        ev1 = torch_npu.npu.Event()
        ev2 = torch_npu.npu.Event(enable_timing=True, blocking=True)
        ev1.record()
        ev2.record()
        ev1.synchronize()
        ev2.synchronize()

    def test_event_sync(self):
        self.framework("event sync", self.ms_event_sync, self.torch_event_sync)

    def ms_event_query(self):
        ev1 = ms.runtime.Event()
        ev1.record()
        ev1.query()

    def torch_event_query(self):

        ev1 = torch_npu.npu.Event()
        ev1.record()
        ev1.query()

    def test_event_query(self):
        self.framework("event query", self.ms_event_query, self.torch_event_query)

    def ms_event_elapsed_time(self):
        start = ms.runtime.Event(enable_timing=True)
        end = ms.runtime.Event(enable_timing=True)
        start.record()
        end.record()
        start.synchronize()
        end.synchronize()
        _ = start.elapsed_time(end)

    def torch_event_elapsed_time(self):
        start = torch_npu.npu.Event(enable_timing=True)
        end = torch_npu.npu.Event(enable_timing=True)
        start.record()
        end.record()
        start.synchronize()
        end.synchronize()
        _ = start.elapsed_time(end)

    def test_event_elapsed_time(self):
        self.framework("event elapsed time", self.ms_event_elapsed_time, self.torch_event_elapsed_time)

    def ms_event_wait(self):
        ev1 = ms.runtime.Event()
        ev1.record()
        ev1.wait()

    def torch_event_wait(self):
        ev1 = torch_npu.npu.Event()
        ev1.record()
        ev1.wait()

    def test_event_wait(self):
        self.framework("event wait", self.ms_event_wait, self.torch_event_wait, 100)

impl = TestStreamApi()
impl.setup_method()
impl.test_event()
impl.test_event_record()
impl.test_event_wait()
impl.test_event_query()
impl.test_event_elapsed_time()
impl.test_event_sync()

print("*" * 10, flush=True)

impl.test_wait_stream()
impl.test_stream()
impl.test_stream_wait_event()
impl.test_stream_record_event()
impl.test_current_stream()
impl.test_default_stream()
impl.test_stream_sync()
impl.test_set_stream()
impl.test_stream_query()
impl.test_stream_ctx()
