package com.mindspore.lite;

import com.mindspore.config.Version;
import java.io.*;
import java.util.logging.Logger;

public class NativeLibrary {
    private static final Logger LOGGER = Logger.getLogger(NativeLibrary.class.toString());

    private static final String GLOG_LIBNAME = "glog";
    private static final String MINDSPORE_LITE_LIBNAME = "mindspore-lite";
    private static final String MINDSPORE_LITE_JNI_LIBNAME = "mindspore-lite-jni";

    public static void load() {
        if (isLibLoaded() || loadLibrary()) {
            LOGGER.info("Native lib has been loaded.");
            return;
        }
        loadLib(makeResourceName("lib" + GLOG_LIBNAME + ".so"));
        loadLib(makeResourceName("lib" + MINDSPORE_LITE_LIBNAME + ".so"));
        loadLib(makeResourceName("lib" + MINDSPORE_LITE_JNI_LIBNAME + ".so"));
    }

    private static boolean isLibLoaded() {
        try {
            Version.version();
        } catch (UnsatisfiedLinkError e) {
            return false;
        }
        return true;
    }

    private static boolean loadLibrary() {
        try {
            System.loadLibrary(GLOG_LIBNAME);
            System.loadLibrary(MINDSPORE_LITE_LIBNAME);
            System.loadLibrary(MINDSPORE_LITE_JNI_LIBNAME);
            LOGGER.info("loadLibrary: success");
            return true;
        } catch (UnsatisfiedLinkError e) {
            LOGGER.warning("tryLoadLibraryFailed: " + e.getMessage());
            return false;
        }
    }

    private static void loadLib(String libResourceName) {
        LOGGER.info("start load libResourceName: " + libResourceName);
        final InputStream libResource = NativeLibrary.class.getClassLoader().getResourceAsStream(libResourceName);
        if (libResource == null) {
            LOGGER.warning(String.format("lib file: %s not exist.", libResourceName));
            return;
        }
        try {
            final File tmpDir = mkTmpDir();
            String libName = libResourceName.substring(libResourceName.lastIndexOf("/") + 1);
            tmpDir.deleteOnExit();

            //copy file to tmpFile
            final File tmpFile = new File(tmpDir.getCanonicalPath(), libName);
            tmpFile.deleteOnExit();
            LOGGER.info(String.format("extract %d bytes to %s", copyLib(libResource, tmpFile), tmpFile));
            System.load(tmpFile.toString());
        } catch (IOException e) {
            throw new UnsatisfiedLinkError(
                    String.format("extract library into tmp file (%s) failed.", e.toString()));
        }
    }

    private static long copyLib(InputStream libResource, File tmpFile) throws IOException{
        try (FileOutputStream outputStream = new FileOutputStream(tmpFile);) {
            // 1MB
            byte[] buffer = new byte[1 << 20];
            long byteCnt = 0;
            int n = 0;
            while ((n = libResource.read(buffer)) >= 0) {
                outputStream.write(buffer, 0, n);
                byteCnt += n;
            }
            return byteCnt;
        } finally {
            libResource.close();
        }
    }


    private static File mkTmpDir() {
        final String MINDSPORE_LITE_LIBS = "mindspore_lite_libs-";
        Long timestamp = System.currentTimeMillis();
        String dirName = MINDSPORE_LITE_LIBS + timestamp + "-";
        for (int i = 0; i < 10; i++) {
            File tmpDir = new File(new File(System.getProperty("java.io.tmpdir")), dirName + i);
            if (tmpDir.mkdir()) {
                return tmpDir;
            }
        }
        throw new IllegalStateException("create tmp dir failed, dirName: " + dirName);
    }

    /**
     * get native lib file name, eg. com/mindspore/lite/linux_x86_64/libmindspore-lite.so
     *
     * @param basename
     * @return
     */
    private static String makeResourceName(String basename) {
        return "com/mindspore/lite/" + String.format("linux_%s/", architecture()) + basename;
    }

    private static String architecture() {
        final String arch = System.getProperty("os.arch").toLowerCase();
        return (arch.equals("amd64")) ? "x86_64" : arch;
    }
}
