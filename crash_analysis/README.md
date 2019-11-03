# reverse-from-coredump
Reverse Execution From CoreDump

## Prerequirement

### libelf / libdisasm

    $ sudo apt-get install libelf1 libelf-dev

library to read and write ELF files

### custome-tailor libdisasm

Now we use a custome-tailor [libdisasm](https://github.com/junxzm1990/libdsiasm). The corresponding installation process is as follows:

```sh
cd libdsiasm
./configure
make
sudo make install

sudo ldconfig
```

### autoconf / automake

    $ sudo apt-get install autoconf automake

## Building

```
$ ./autogen.sh
$ ./configure
$ make
```

## Usage

    $ ./src/reverse case_path

**Make sure binary file and all the library files are in the case_path**
