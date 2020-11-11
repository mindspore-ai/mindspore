package com.mindspore.himindspore.mvp;

import com.mindspore.himindspore.net.FileDownLoadObserver;
import com.mindspore.himindspore.net.UpdateInfoBean;

import java.io.File;

public interface MainContract {

    interface View {
        void showUpdateResult(UpdateInfoBean object);

        void showFail(String s);

    }

    interface Presenter {
        void getUpdateInfo();

        void downloadApk(String destDir, String fileName, FileDownLoadObserver<File> fileDownLoadObserver);
    }
}
