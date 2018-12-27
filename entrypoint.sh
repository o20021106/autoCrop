#!/bin/bash
set -e
export PATH=$PATH:model_files/binary/
echo $PATH
exec "$@"
