#! /bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"

# Remove previous build
rm helloworld
# Testing compile
echo "Compiling helloworld.c with dpu-upmem-dpurte-clang.."
dpu-upmem-dpurte-clang -o helloworld helloworld.c

echo "Compilation finished."

# Testing execution
echo "Running helloworld..."
dpu-lldb --batch helloworld -o "process launch"

EXITCODE=$?   # captures dpu-lldb's own exit code
echo "dpu-lldb exited with code $EXITCODE"

rm helloworld

if [ $EXITCODE -eq 0 ]; then
    echo "You have successfully run the helloworld test!"
    echo -e "\nSDK is set up correctly."
else
    echo "dpu-lldb failed!"
    echo -e "Test failed, SDK not set up correctly."
fi
