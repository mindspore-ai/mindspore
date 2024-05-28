[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
![Build Status](https://github.com/megastep/makeself/workflows/CI/badge.svg)

# makeself - Make self-extractable archives on Unix

[makeself.sh][1] is a small shell script that generates a self-extractable
compressed tar archive from a directory. The resulting file appears as a shell script
(many of those have a **.run** suffix), and can be launched as is. The archive
will then uncompress itself to a temporary directory and an optional arbitrary
command will be executed (for example an installation script). This is pretty
similar to archives generated with WinZip Self-Extractor in the Windows world.
Makeself archives also include checksums for integrity self-validation (CRC
and/or MD5/SHA256 checksums).

The makeself.sh script itself is used only to create the archives from a
directory of files. The resultant archive is actually a compressed (using
gzip, bzip2, or compress) TAR archive, with a small shell script stub at the
beginning. This small stub performs all the steps of extracting the files,
running the embedded command, and removing the temporary files when done.
All the user has to do to install the software contained in such an
archive is to "run" the archive, i.e **sh nice-software.run**. I recommend
using the ".run" (which was introduced by some Makeself archives released by
Loki Software) or ".sh" suffix for such archives not to confuse the users,
so that they will know they are actually shell scripts (with quite a lot of binary data
attached to them though!).

I am trying to keep the code of this script as portable as possible, i.e it is
not relying on any bash-specific features and only calls commands that are
installed on any functioning UNIX-compatible system. This script as well as
the archives it generates should run on any Unix flavor, with any compatible
Bourne shell, provided of course that the compression programs are available.

As of version 2.1, Makeself has been rewritten and tested on the following
platforms :

  * Linux (all distributions)
  * Sun Solaris (8 and above)
  * HP-UX (tested on 11.0 and 11i on HPPA RISC)
  * SCO OpenUnix and OpenServer
  * IBM AIX 5.1L
  * macOS (Darwin)
  * SGI IRIX 6.5
  * FreeBSD
  * UnicOS / Cray
  * Cygwin (Windows)

If you successfully run Makeself and/or archives created with it on another
system, then please [let me know][2]!

Examples of publicly available archives made using makeself are :

  * Game patches and installers for [Id Software][3] games like Quake 3 for Linux or Return To Castle Wolfenstein ;
  * All game patches released by [Loki Software][4] for the Linux version of popular games ;
  * The [nVidia drivers][5] for Linux
  * The installer for the Linux version of [Google Earth][6]
  * The [VirtualBox][7] installers for Linux
  * The [Makeself][1] distribution itself ;-)
  * and countless others...

**Important note for Apache users:** By default, most Web servers will think that Makeself archives are regular text files and thus they may show up as text in a Web browser. The correct way to prevent this is to add a MIME type for this file format, like so (in httpd.conf) :

`AddType application/x-makeself .run`

**Important note for certain GNU/Linux distributions:** Archives created with Makeself prior to v2.1.2 were using an old syntax for the _head_ and _tail_ Unix commands that is being progressively obsoleted in their GNU forms. Therefore you may have problems uncompressing some of these archives. A workaround for this is to set the environment variable $_POSIX2_VERSION to enable the old syntax, i.e. :

`export _POSIX2_VERSION=199209`

## Usage

The syntax of makeself is the following:

```
makeself.sh [args] archive_dir file_name label startup_script [script_args]
```

  * _args_ are optional options for Makeself. The available ones are :

    * **`--version`** : Prints the version number on stdout, then exits immediately
    * **`--gzip`** : Use gzip for compression (the default on platforms on which gzip is commonly available, like Linux)
    * **`--bzip2`** : Use bzip2 instead of gzip for better compression. The bzip2 command must be available in the command path. It is recommended that the archive prefix be set to something like '.bz2.run', so that potential users know that they'll need bzip2 to extract it.
    * **`--pbzip2`** : Use pbzip2 instead of gzip for better and faster compression on machines having multiple CPUs. The pbzip2 command must be available in the command path. It is recommended that the archive prefix be set to something like '.bz2.run', so that potential users know that they'll need bzip2 to extract it.
    * **`--xz`** : Use xz instead of gzip for better compression. The xz command must be available in the command path. It is recommended that the archive prefix be set to something like '.xz.run' for the archive, so that potential users know that they'll need xz to extract it.
    * **`--lzo`** : Use lzop instead of gzip for better compression. The lzop command must be available in the command path. It is recommended that the archive prefix be set to something like `.lzo.run` for the archive, so that potential users know that they'll need lzop to extract it.
    * **`--lz4`** : Use lz4 instead of gzip for better compression. The lz4 command must be available in the command path. It is recommended that the archive prefix be set to something like '.lz4.run' for the archive, so that potential users know that they'll need lz4 to extract it.
    * **`--zstd`** : Use zstd instead of gzip for better compression. The zstd command must be available in the command path. It is recommended that the archive prefix be set to something like '.zstd.run' for the archive, so that potential users know that they'll need zstd to extract it.
    * **`--pigz`** : Use pigz for compression.
    * **`--base64`** : Encode the archive to ASCII in Base64 format instead of compressing (base64 command required).
    * **`--gpg-encrypt`** : Encrypt the archive using `gpg -ac -z $COMPRESS_LEVEL`. This will prompt for a password to encrypt with. Assumes that potential users have `gpg` installed.
    * **`--ssl-encrypt`** : Encrypt the archive using `openssl aes-256-cbc -a -salt`. This will prompt for a password to encrypt with. Assumes that the potential users have the OpenSSL tools installed.
    * **`--compress`** : Use the UNIX `compress` command to compress the data. This should be the default on all platforms that don't have gzip available.
    * **`--nocomp`** : Do not use any compression for the archive, which will then be an uncompressed TAR.
    * **`--complevel`** : Specify the compression level for gzip, bzip2, pbzip2, zstd, xz, lzo or lz4. (defaults to 9)
    * **`--threads`** : Specify the number of threads to be used by compressors that support parallelization. Omit to use compressor's default. Most useful (and required) for opting into xz's threading, usually with `--threads=0` for all available cores. pbzip2 and pigz are parallel by default, and setting this value allows limiting the number of threads they use.
    * **`--notemp`** : The generated archive will not extract the files to a temporary directory, but in a new directory created in the current directory. This is better to distribute software packages that may extract and compile by themselves (i.e. launch the compilation through the embedded script).
    * **`--current`** : Files will be extracted to the current directory, instead of in a subdirectory. This option implies `--notemp` above.
    * **`--follow`** : Follow the symbolic links inside of the archive directory, i.e. store the files that are being pointed to instead of the links themselves.
    * **`--append`** _(new in 2.1.x)_: Append data to an existing archive, instead of creating a new one. In this mode, the settings from the original archive are reused (compression type, label, embedded script), and thus don't need to be specified again on the command line.
    * **`--header`** : Makeself uses a separate file to store the header stub, called `makeself-header.sh`. By default, it is assumed that it is stored in the same location as makeself.sh. This option can be used to specify its actual location if it is stored someplace else.
    * **`--cleanup`** : Specify a script that is run when execution is interrupted or finishes successfully. The script is executed with the same environment and initial `script_args` as `startup_script`. 
    * **`--copy`** : Upon extraction, the archive will first extract itself to a temporary directory. The main application of this is to allow self-contained installers stored in a Makeself archive on a CD, when the installer program will later need to unmount the CD and allow a new one to be inserted. This prevents "Filesystem busy" errors for installers that span multiple CDs.
    * **`--nox11`** : Disable the automatic spawning of a new terminal in X11.
    * **`--nowait`** : When executed from a new X11 terminal, disable the user prompt at the end of the script execution.
    * **`--nomd5`** and **`--nocrc`** : Disable the creation of a MD5 / CRC checksum for the archive. This speeds up the extraction process if integrity checking is not necessary.
    * **`--sha256`** : Adds a SHA256 checksum for the archive. This is in addition to the MD5 / CRC checksums unless `--nomd5` is also used.
    * **`--lsm` _file_** : Provide and LSM file to makeself, that will be embedded in the generated archive. LSM files are describing a software package in a way that is easily parseable. The LSM entry can then be later retrieved using the `--lsm` argument to the archive. An example of a LSM file is provided with Makeself.
    * **`--tar-format opt`** : Specify the tar archive format (default is ustar); you may use any value accepted by your tar command (such as posix, v7, etc).
    * **`--tar-extra opt`** : Append more options to the tar command line.

        For instance, in order to exclude the `.git` directory from the packaged archive directory using the GNU `tar`, one can use `makeself.sh --tar-extra "--exclude=.git" ...`

    * **`--keep-umask`** : Keep the umask set to shell default, rather than overriding when executing self-extracting archive.
    * **`--packaging-date date`** : Use provided string as the packaging date instead of the current date.
    * **`--license`** : Append a license file.
    * **`--nooverwrite`** : Do not extract the archive if the specified target directory already exists.
    * **`--help-header file`** : Add a header to the archive's `--help` output.
  * `archive_dir` is the name of the directory that contains the files to be archived
  * `file_name` is the name of the archive to be created
  * `label` is an arbitrary text string describing the package. It will be displayed while extracting the files.
  * `startup_script` is the command to be executed _from within_ the directory of extracted files. Thus, if you wish to execute a program contained in this directory, you must prefix your command with `./`. For example, `./program` will be fine. The `script_args` are additional arguments for this command.

Here is an example, assuming the user has a package image stored in a **/home/joe/mysoft**, and he wants to generate a self-extracting package named
**mysoft.sh**, which will launch the "setup" script initially stored in /home/joe/mysoft :

`makeself.sh /home/joe/mysoft mysoft.sh "Joe's Nice Software Package" ./setup
`

Here is also how I created the [makeself.run][9] archive which contains the Makeself distribution :

`makeself.sh --notemp makeself makeself.run "Makeself by Stephane Peter" echo "Makeself has extracted itself" `

Archives generated with Makeself can be passed the following arguments:

  * **`--keep`** : Prevent the files to be extracted in a temporary directory that will be removed after the embedded script's execution. The files will then be extracted in the current working directory and will stay here until you remove them.
  * **`--verbose`** : Will prompt the user before executing the embedded command
  * **`--target dir`** : Allows to extract the archive in an arbitrary place.
  * **`--nox11`** : Do not spawn a X11 terminal.
  * **`--confirm`** : Prompt the user for confirmation before running the embedded command.
  * **`--info`** : Print out general information about the archive (does not extract).
  * **`--lsm`** : Print out the LSM entry, if it is present.
  * **`--list`** : List the files in the archive.
  * **`--check`** : Check the archive for integrity using the embedded checksums. Does not extract the archive.
  * **`--nochown`** : By default, a `chown -R` command is run on the target directory after extraction, so that all files belong to the current user. This is mostly needed if you are running as root, as tar will then try to recreate the initial user ownerships. You may disable this behavior with this flag.
  * **`--tar`** : Run the tar command on the contents of the archive, using the following arguments as parameter for the command.
  * **`--noexec`** : Do not run the embedded script after extraction.
  * **`--noexec-cleanup`** : Do not run the embedded cleanup script.
  * **`--nodiskspace`** : Do not check for available disk space before attempting to extract.
  * **`--cleanup-args`** : Specify arguments to be passed to the cleanup script. Wrap value in quotes to specify multiple arguments.

Any subsequent arguments to the archive will be passed as additional arguments to the embedded command. You must explicitly use the `--` special command-line construct before any such options to make sure that Makeself will not try to interpret them.

## Startup Script

The startup script must be a regular Shell script. 

Within the startup script, you can use the `$USER_PWD` variable to get the path of the folder from which the self-extracting script is executed. This is especially useful to access files that are located in the same folder as the script, as shown in the example below. 

`my-self-extracting-script.sh --fooBarFileParameter foo.bar`

## Building and Testing

Clone the git repo and execute `git submodule update --init --recursive` to obtain all submodules.

* To make a release: `make`
* To run all tests:  `make test`

## Maven Usage

Makeself is now supported by the following maven plugin [makeself-maven-plugin](https://github.com/hazendaz/makeself-maven-plugin).  Please refer to project for usage and report any bugs in regards to maven plugin on that project.

## License

Makeself itself is covered by the [GNU General Public License][8] (GPL) version 2 and above. Archives generated by Makeself don't have to be placed under this license (although I encourage it ;-)), since the archive itself is merely data for Makeself.

## Contributing

I will gladly consider merging your pull requests on the [GitHub][10] repository. However, please keep the following in mind:

  * One of the main purposes of Makeself is portability. Do not submit patches that will break supported platforms. The more platform-agnostic, the better.
  * Please explain clearly what the purpose of the patch is, and how you achieved it.

## Download

Get the latest official distribution [here][9] (version 2.4.2).

The latest development version can be grabbed from [GitHub][10]. Feel free to submit any patches there through the fork and pull request process.

## Version history

  * **v1.0:** Initial public release
  * **v1.1:** The archive can be passed parameters that will be passed on to the embedded script, thanks to John C. Quillan
  * **v1.2:** Cosmetic updates, support for bzip2 compression and non-temporary archives. Many ideas thanks to Francois Petitjean.
  * **v1.3:** More patches from Bjarni R. Einarsson and Francois Petitjean: Support for no compression (`--nocomp`), script is no longer mandatory, automatic launch in an xterm, optional verbose output, and -target archive option to indicate where to extract the files.
  * **v1.4:** Many patches from Francois Petitjean: improved UNIX compatibility, automatic integrity checking, support of LSM files to get info on the package at run time..
  * **v1.5.x:** A lot of bugfixes, and many other patches, including automatic verification through the usage of checksums. Version 1.5.5 was the stable release for a long time, even though the Web page didn't get updated ;-). Makeself was also officially made a part of the [Loki Setup installer][11], and its source is being maintained as part of this package.
  * **v2.0:** Complete internal rewrite of Makeself. The command-line parsing was vastly improved, the overall maintenance of the package was greatly improved by separating the stub from makeself.sh. Also Makeself was ported and tested to a variety of Unix platforms.
  * **v2.0.1:** First public release of the new 2.0 branch. Prior versions are officially obsoleted. This release introduced the `--copy` argument that was introduced in response to a need for the [UT2K3][12] Linux installer.
  * **v2.1.0:** Big change : Makeself can now support multiple embedded tarballs, each stored separately with their own checksums. An existing archive can be updated with the `--append` flag. Checksums are also better managed, and the `--nochown` option for archives appeared.
  * **v2.1.1:** Fixes related to the Unix compression (compress command). Some Linux distributions made the insane choice to make it unavailable, even though gzip is capable of uncompressing these files, plus some more bugfixes in the extraction and checksum code.
  * **v2.1.2:** Some bug fixes. Use head -n to avoid problems with POSIX conformance.
  * **v2.1.3:** Bug fixes with the command line when spawning terminals. Added `--tar`, `--noexec` for archives. Added `--nomd5` and `--nocrc` to avoid creating checksums in archives. The embedded script is now run through "eval". The `--info` output now includes the command used to create the archive. A man page was contributed by Bartosz Fenski.
  * **v2.1.4:** Fixed `--info` output. Generate random directory name when extracting files to . to avoid problems. Better handling of errors with wrong permissions for the directory containing the files. Avoid some race conditions, Unset the $CDPATH variable to avoid problems if it is set. Better handling of dot files in the archive directory.
  * **v2.1.5:** Made the md5sum detection consistent with the header code. Check for the presence of the archive directory. Added `--encrypt` for symmetric encryption through gpg (Eric Windisch). Added support for the digest command on Solaris 10 for MD5 checksums. Check for available disk space before extracting to the target directory (Andreas Schweitzer). Allow extraction to run asynchronously (patch by Peter Hatch). Use file descriptors internally to avoid error messages (patch by Kay Tiong Khoo).
  * **v2.1.6:** Replaced one dot per file progress with a realtime progress percentage and a spinning cursor. Added `--noprogress` to prevent showing the progress during the decompression. Added `--target` dir to allow extracting directly to a target directory. (Guy Baconniere)
  * **v2.2.0:** First major new release in years! Includes many bugfixes and user contributions. Please look at the [project page on Github][10] for all the details.
  * **v2.3.0:** Support for archive encryption via GPG or OpenSSL. Added LZO and LZ4 compression support. Options to set the packaging date and stop the umask from being overriden. Optionally ignore check for available disk space when extracting. New option to check for root permissions before extracting.
  * **v2.3.1:** Various compatibility updates. Added unit tests for Travis CI in the GitHub repo. New `--tar-extra`, `--untar-extra`, `--gpg-extra`, `--gpg-asymmetric-encrypt-sign` options.
  * **v2.4.0:** Added optional support for SHA256 archive integrity checksums.
  * **v2.4.2:** New --cleanup and --cleanup-args arguments for cleanup scripts. Added threading support for supported compressors. Now supports zstd compression.
  * **v2.4.3:** Make explicit POSIX tar archives for increased compatibility.
  * **v2.4.4:** Fixed various compatibility issues (no longer use POSIX tar archives), Github Actions to check on Solaris and FreeBSD.
  * **v2.4.5:** Added `--tar-format` option to set the tar archive format (default is ustar)

## Links

  * Check out the ["Loki Setup"][11] installer, used to install many Linux games and other applications, and of which I am the co-author. Since the demise of Loki, I am now the official maintainer of the project, and it is now being hosted here on GitHub.
  * Bjarni R. Einarsson also wrote the **setup.sh** installer script, inspired by Makeself. [Check it out !][14]

## Contact

This script was written by [Stéphane Peter][15] (megastep at megastep.org). Any enhancements and suggestions are welcome.

Contributions were included from John C. Quillan, Bjarni R. Einarsson,
Francois Petitjean, Ryan C. Gordon, and many contributors on GitHub. If you think I forgot
your name, don't hesitate to contact me.

This project is now hosted on GitHub. Feel free to submit patches and bug reports on the [project page][10].

* * *

[Stephane Peter][2]

   [1]: http://makeself.io/
   [2]: mailto:megastep@megastep.org
   [3]: http://www.idsoftware.com/
   [4]: http://www.lokigames.com/products/myth2/updates.php3
   [5]: http://www.nvidia.com/
   [6]: http://earth.google.com/
   [7]: http://www.virtualbox.org/
   [8]: http://www.gnu.org/copyleft/gpl.html
   [9]: https://github.com/megastep/makeself/releases/download/release-2.4.5/makeself-2.4.5.run
   [10]: https://github.com/megastep/makeself
   [11]: https://github.com/megastep/loki_setup/
   [12]: http://www.unrealtournament2003.com/
   [13]: http://www.icculus.org/
   [14]: http://bre.klaki.net/programs/setup.sh/
   [15]: https://stephanepeter.com/
