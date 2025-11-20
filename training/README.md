Usage:
1. Install the dependencies listed in `requirements-train.txt`.
2. Configure these environment variables on every node: `MLP_GPU`, `MLP_WORKER_NUM`, `MLP_ROLE_INDEX`, `MLP_WORKER_0_HOST`, `MLP_WORKER_0_PORT`.
3. On the head node, run `bash scripts/setup.sh` to start Ray.
4. On the head node, run `bash scripts/start.sh` to start training.

