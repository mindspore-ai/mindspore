package com.mindspore.lite.context;

public class Context {
    private long contextPtr;

    public Context() {
        this.contextPtr = 0;
    }

    public long getContextPtr() {
        return contextPtr;
    }

    public void setContextPtr(long contextPtr) {
        this.contextPtr = contextPtr;
    }

    public boolean init(int deviceType, int threadNum, int cpuBindMode) {
        this.contextPtr = createContext(deviceType, threadNum, cpuBindMode);
        return this.contextPtr != 0;
    }

    public boolean init(int deviceType, int threadNum) {
        return init(deviceType, threadNum, CpuBindMode.MID_CPU);
    }

    public boolean init(int deviceType) {
        return init(deviceType, 2);
    }

    public boolean init() {
        return init(DeviceType.DT_CPU);
    }

    public void free() {
        this.free(this.contextPtr);
        this.contextPtr = 0;
    }

    private native long createContext(int deviceType, int threadNum, int cpuBindMode);

    private native void free(long contextPtr);
}
