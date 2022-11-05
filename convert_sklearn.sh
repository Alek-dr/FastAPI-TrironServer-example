#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if [ $# -ne 1 ] || [ ! -f $1 ]
then
  echo "USAGE: convert_sklearn PKL_FILE"
else
  out_file="$(dirname $1)/checkpoint.tl"
  source venv/bin/activate
  python3 -m treelite.serialize --input-model "$1" \
    --input-model-type sklearn_pkl --output-checkpoint "$out_file"
fi