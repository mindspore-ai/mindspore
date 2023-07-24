# workspace-root
# --- mindir
#     --- model
#     --- input
#     --- output
# --- caffe
#     --- model
#     --- input
#     --- output
# --- onnx
#     --- model
#     --- input
#     --- output
# --- tf
#     --- model
#     --- input
#     --- output
# --- tflite
#     --- model
#     --- input
#     --- output

from context import Context
from test_manager import TestManager

if __name__ == "__main__":
    context = Context.instance()
    context.init()
    parser = TestManager.parse(context.config_file)

    ret_code = 0
    if context.mode.convert:
        if not parser.test_convert():
            ret_code = -1
            if context.exit_on_failed:
                parser.summary()
                exit(ret_code)
    if context.mode.performance:
        if not parser.test_bench_performance():
            ret_code = -1
            if context.exit_on_failed:
                parser.summary()
                exit(ret_code)
    if context.mode.accuracy:
        if not parser.test_bench_acc():
            ret_code = -1
            if context.exit_on_failed:
                parser.summary()
                exit(ret_code)
    parser.summary()
    exit(ret_code)
