# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
from pathlib import Path

from defs.conftest import llm_models_root
from defs.trt_test_alternative import check_call, check_output

llama_path = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"


def trtllm_bench_prepare_dataset(llm_root, llm_venv):
    dataset_tool = Path(llm_root, "benchmarks", "cpp", "prepare_dataset.py")
    dataset_path = tempfile.NamedTemporaryFile(delete=False).name

    command = [
        f"{dataset_tool.resolve()}",
        "--stdout",
        "--tokenizer",
        f"{llama_path}",
        "token-norm-dist",
        "--input-mean",
        "128",
        "--output-mean",
        "128",
        "--input-stdev",
        "0",
        "--output-stdev",
        "0",
        "--num-requests",
        "10",
    ]
    print(f"command: {' '.join(command)}")
    dataset_output = llm_venv.run_cmd(
        command,
        caller=check_output,
    )

    with open(dataset_path, "w") as dataset:
        dataset.write(dataset_output)

    return dataset_path


def test_llmapi_trtllm_bench_pytorch(llm_root, llm_venv):

    command = [
        "mpirun",
        "-n",
        "2",
        "trtllm-llmapi-launch",
        "trtllm-bench",
        "--model",
        "meta-llama/Llama-3.1-8B",
        "--model_path",
        llama_path,
        "throughput",
        "--dataset",
        trtllm_bench_prepare_dataset(llm_root, llm_venv),
        "--tp",
        "2",
        "--backend",
        "pytorch",
    ]
    print(f"command: {' '.join(command)}")

    check_call(command, env=os.environ.copy())


# TODO add test for TRT backend
