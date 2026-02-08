# pre-push快速指引

<!-- TOC -->

- [pre-push快速指引](#pre-push快速指引)
    - [本地使用pre-push步骤](#本地使用pre-push步骤)
    - [附：常见问题QA](#附常见问题qa)
    - [附：手动安装代码检查工具](#附手动安装代码检查工具)
        - [Windows环境](#windows环境)
        - [Linux环境](#linux环境)
        - [Mac环境](#mac环境)
    - [附：工具版本建议](#附工具版本建议)

<!-- TOC -->

## 本地使用pre-push步骤

1. 确认环境

  确认本地环境已经安装Git工具、Python（**python --version命令打印的版本信息是3.9、3.10或3.11版本**）、pip命令。

2. 使用脚本安装代码检查工具

  在`mindspore/`目录下执行以下命令进行自动安装：

  ```bash
  cd scripts/pre_commit
  bash install_generic_tools.sh
  bash install_system_specific_tools.sh
  ```

  `install_generic_tools.sh`安装的是`cmakelint`、`codespell`、`cpplint`、`lizard`、`pylint`、`clang-format`工具，`install_system_specific_tools.sh`安装的是`shellcheck`工具。

  **注意**：

- 在Linux或者Mac环境下执行install_system_specific_tools.sh时涉及sudo命令，请确保执行用户具有sudo权限。
- 由于本地环境各不相同，在安装过程中可能出现某些工具安装失败或者安装的工具版本较低的情况，可参考[手动安装](##附手动安装代码检查工具)部分重新安装。
- 不同环境下，我们对每个工具的安装版本有不同的建议，详情请参考[工具版本建议](##附工具版本建议)部分，但是只要不低于CI门禁上的版本都是可以正常使用的。

3. 使用pre-push

    （1）拉取master分支最新代码。

    （2）配置git的hooks路径为pre-push所在的目录。pre-push文件位于`mindspore/scripts/pre_commit/githooks/pre-push`，因此，在`mindspore/`目录下执行：

    ```bash
    git config core.hooksPath scripts/pre_commit/githooks
    ```

    **注意**：`core.hooksPath`的参数是pre-push所在的目录，路径上不可以包含pre-push。

    （3）运行pre-push

    pre-push不用手动执行，每次执行`git push`推送代码会自动触发pre-push对本次推送的代码进行扫描。

    （4）查看执行结果

    pre-push执行结束时会输出`Total error number is xxx`提示总共扫描出的告警数量。如果告警数量为0，代码将继续推送到仓库；反之则会拒绝推送。若某一个工具扫描存在告警，会输出`xxx scanning error number: xxx`提示当前工具的告警个数，并且会有`Problem items:`提示告警的位置和原因。

    **（5）绕过pre-push**

    如果希望本次推送的代码不被扫描，或者告警的位置是其他人的代码，使用`git push --no-verify`命令推送代码可绕过pre-push检查。

## 附：常见问题QA

- **Q**：为什么本地扫描结果与CI门禁不一致？

  **A**：在不同的环境上扫描结果不尽相同，本地无法保证与CI环境一致，因此本地扫描结果仅供参考。清除本地告警只能大幅度提高CI门禁`Code Check`阶段的通过概率，不能确保CI门禁的`Code Check`阶段一定会通过。

- **Q**：当扫描出来的告警只会在本地出现，怎么让本地不再出现同样的告警呢？

  **A**：会出现上述情况的是`cpplint`、`pylint`、`lizard`这三个工具，这三个工具在`.jenkins/check/config`下提供了白名单，可以将只会在本地出现的告警添加到对应的白名单文件中进行屏蔽。**白名单文件修改后请不到推送到CI仓库**。

- **Q**：使用自动安装方式安装工具时，为什么有些工具没有安装或者安装的工具版本较低？

  **A**：（1）为了不影响原有的环境，脚本中使用常规安装命令安装系统推荐的版本，由于系统版本的不同，有的工具会出现无法安装或者推荐安装的工具版本较低，请自行在官网下载安装包进行解压安装。

  （2）Git工具自带tab工具，无需安装，所以安装过程不涉及tab工具。

  （3）Windows的shellcheck工具扫描结果不具有参考价值，安装脚本中不包含Windows的shellcheck工具，如果需要扫描shellcheck，请在Linux或者Mac环境推送代码。

- **Q**：没有成功安装所有的工具可以使用pre-push吗？

  **A**：下载其中的任何几个工具都可以正常使用pre-push。pre-push会检查已安装哪些工具，用已安装的工具对代码进行扫描，没有安装的工具则跳过。

## 附：手动安装代码检查工具

部分工具使用脚本无法成功安装，需要自己手动安装。

### Windows环境

Windows环境的命令请在`git bash`窗口执行。

1. clang-format

   （1）安装clang-format：

   ```bash
   pip install --upgrade --force-reinstall clang-format
   ```

   （2）查看版本信息：

   ```bash
   clang-format --version
   ```

2. cmakelint

   （1）安装cmakelint：

   ```bash
   pip install --upgrade --force-reinstall cmakelint
   ```

   （2）查看版本信息：

   ```bash
   cmakelint --version
   ```

3. codespell

   （1）安装codespell：

   ```bash
   pip install --upgrade --force-reinstall codespell
   ```

   （2）查看版本信息：

   ```bash
   codespell --version
   ```

4. cpplint

   （1）安装cpplint：

   ```bash
   pip install --upgrade --force-reinstall cpplint
   ```

   （2）查看版本信息：

   ```bash
   cpplint --version
   ```

5. lizard

   （1）安装lizard：

   ```bash
   pip install --upgrade --force-reinstall lizard
   ```

   （2）查看版本信息：

   ```bash
   lizard --version
   ```

6. pylint

   （1）安装pylint：

   ```bash
   pip install pylint==3.3.7
   ```

   （2）查看版本信息：

   ```bash
   pylint --version
   ```

7. shellcheck

   Windows的shellcheck工具扫描结果不具有参考价值，建议不安装。

8. tab

   Git工具自带tab工具，不需要单独安装tab。

### Linux环境

Linux的发行版本众多，无法兼容所有的发行版本，本文以CentOS x86_64为例。

1. clang-format（[同Windows环境](#windows环境)）

2. cmakelint（[同Windows环境](#windows环境)）

3. codespell（[同Windows环境](#windows环境)）

4. cpplint（[同Windows环境](#windows环境)）

5. lizard（[同Windows环境](#windows环境)）

6. pylint（[同Windows环境](#windows环境)）

7. shellcheck

   （1）下载shellcheck安装包到`/tmp`目录：

   ```bash
   cd /tmp
   wget https://github.com/koalaman/shellcheck/releases/download/v0.8.0/shellcheck-v0.8.0.linux.x86_64.tar.xz --no-check-certificate
   ```

   （2）解压安装shellcheck工具：

   ```bash
   tar -xf shellcheck-v0.8.0.linux.x86_64.tar.xz
   rm -f /usr/bin/shellcheck
   mv /tmp/shellcheck-0.8.0/shellcheck /usr/bin/shellcheck
   chmod 755 /usr/bin/shellcheck
   rm -rf /tmp/shellcheck-v0.8.0
   rm -f /tmp/shellcheck-v0.8.0.linux.x86_64.tar.xz
   ```

   （3）查看版本信息：

   ```bash
   shellcheck --version
   ```

8. tab

   Git工具自带tab工具，不需要单独安装tab。

### Mac环境

1. clang-format（[同Windows环境](#windows环境)）

2. cmakelint（[同Windows环境](#windows环境)）

3. codespell（[同Windows环境](#windows环境)）

4. cpplint（[同Windows环境](#windows环境)）

5. lizard（[同Windows环境](#windows环境)）

6. pylint（[同Windows环境](#windows环境)）

7. shellcheck

   （1）安装shellcheck：

   ```bash
   brew install shellcheck
   ```

   （2）添加到环境变量：

   ```bash
   brew link --overwrite shellcheck
   ```

   （3）查看版本信息：

   ```bash
   shellcheck --version
   ```

8. tab

   Git工具自带tab工具，不需要单独安装tab。

## 附：工具版本建议

|   工具名称   | CI门禁版本 | 最新版本 | Windows |  Linux  |   Mac   |
| :----------: | :--------: | :------: | :-----: | :-----: | :-----: |
| clang-format |   18.1.8    |  21.1.4  |  18.1.8  | >=18.1.0 | >=18.1.0 |
|  cmakelint   |   1.4.1    |  1.4.3   |  1.4.2  |  1.4.2  |  1.4.2  |
|  codespell   |   2.0.0    |  2.4.1   |  2.1.0  |  2.1.0  |  2.1.0  |
|   cpplint    |   2.0.2    |  2.0.2   |  2.0.2  |  2.0.2  |  2.0.2  |
|    lizard    |   1.17.19   | 1.18.0  | 1.17.10 | 1.17.10 | 1.17.10 |
|    pylint    |   3.3.7    |  4.0.2  |  3.3.7  |  3.3.7  |  3.3.7  |
|  shellcheck  |   0.7.1    |  0.11.0   |    —    |  0.8.0  |  0.8.0  |
|     tab      |     —      |    —     |    —    |    —    |    —    |
