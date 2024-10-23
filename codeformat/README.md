<!--
 * @Author: thuaj@connect.ust.hk
 * @Date: 2024-10-23 13:42:00
 * @LastEditTime: 2024-10-23 14:33:24
 * @Description: code format I used for my projects python and c++
 * Copyright (c) 2024 by thuaj@connect.ust.hk, All Rights Reserved. 
-->

# code format
## C++ 
I use clang-format and follow [google code style](https://google.github.io/styleguide/cppguide.html). An auto format pipeline refer to [ethz-asl linter](https://github.com/ethz-asl/linter).
## set up
```
# install dependencies
sudo apt install clang-format
pip install pyyaml requests yapf pylint
# Download the linter
cd <linter_dest>
git clone git@github.com:ethz-asl/linter.git 
cd linter
echo ". $(realpath setup_linter.sh)" >> ~/.bashrc
echo ". $(realpath setup_linter.sh)" >> ~/.zshrc
bash or zsh
cd <project_dir>
init_linter_git_hooks
```
- if using python3, then replace <linter_dest>/cpplint.py with [cpplint.py](./cpplint.py).

- put [.clang-format](./.clang-format) file in the root directory of your project.

## usage
Checking an entire repository.
```
cd <project_dir>
linter_check_all
```


Disable Linter Functionalities for a Specific Line

```
code ... // NOLINT

// clang-format off
...
// clang-format on   

```
## uninstall
```
cd <project_dir>
init_linter_git_hooks --remove
```
