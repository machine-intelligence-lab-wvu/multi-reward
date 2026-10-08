"""
HuggingFace Hub push callback — pushes checkpoints every N steps to separate branches.

Each checkpoint is pushed to a unique branch: step-50, step-100, step-150, etc.
This lets you differentiate checkpoints across runs and steps.

HF Repo structure:
  shourovj/multi-reward-grpo
    ├── main branch           ← final model (pushed at end of training)
    ├── step-50 branch        ← checkpoint at step 50
    ├── step-100 branch       ← checkpoint at step 100
    ├── step-150 branch       ← checkpoint at step 150
    └── ...

To evaluate a specific checkpoint from your local machine:
  python evaluation/eval_checkpoint.py \
      --checkpoint shourovj/multi-reward-grpo \
      --revision step-100 \
      --dataset math500

Usage in YAML config:
  push_to_hub: true
  hub_model_id: username/repo-name
  hub_private_repo: true
  hub_push_every_n_steps: 50
  run_name: my_experiment          # used as prefix for branch names

Or via CLI:
  --push_to_hub true --hub_model_id username/repo --hub_push_every_n_steps 50
"""

import logging
import os

from transformers import TrainerCallback
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)


class HubPushCallback(TrainerCallback):
    """Push checkpoints to HuggingFace Hub every N training steps.

    Each push goes to a unique branch (e.g., step-50, step-100).
    Decoupled from local save_steps — Hub push interval is independent.
    """

    def __init__(self, push_every_n_steps: int = 50, run_name: str = None):
        self.push_every_n_steps = push_every_n_steps
        self.run_name = run_name  # optional prefix for branch names
        self._last_pushed_step = -1
        self._api = HfApi()

    def _get_branch_name(self, step):
        """Generate branch name for this checkpoint."""
        if self.run_name:
            return f"{self.run_name}-step-{step}"
        return f"step-{step}"

    def _ensure_branch(self, repo_id, branch):
        """Create branch on HF repo if it doesn't exist."""
        try:
            self._api.create_branch(repo_id, branch=branch, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create branch '{branch}': {e}")

    def on_step_end(self, args, state, control, **kwargs):
        """Check if we should push to Hub at this step."""
        if not args.push_to_hub:
            return

        step = state.global_step
        if step > 0 and step % self.push_every_n_steps == 0 and step != self._last_pushed_step:
            control.should_save = True
            self._should_push_after_save = True
            self._last_pushed_step = step

    def on_save(self, args, state, control, **kwargs):
        """After a local save, push to Hub on a unique branch."""
        if not args.push_to_hub:
            return

        step = state.global_step
        should_push = getattr(self, "_should_push_after_save", False)

        if step > 0 and step % self.push_every_n_steps == 0:
            should_push = True

        if not should_push:
            return

        self._should_push_after_save = False
        repo_id = args.hub_model_id
        branch = self._get_branch_name(step)

        try:
            # Find the local checkpoint directory
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
            if not os.path.isdir(checkpoint_dir):
                # Try the latest checkpoint in output_dir
                checkpoint_dir = args.output_dir

            logger.info(f"Pushing step {step} to Hub: {repo_id} (branch: {branch})")

            # Ensure repo exists
            self._api.create_repo(repo_id, private=True, exist_ok=True)

            # Create branch
            self._ensure_branch(repo_id, branch)

            # Upload checkpoint folder to the branch
            self._api.upload_folder(
                folder_path=checkpoint_dir,
                repo_id=repo_id,
                revision=branch,
                commit_message=f"Checkpoint step {step}",
            )

            logger.info(f"Pushed step {step} → {repo_id} (branch: {branch})")

        except Exception as e:
            logger.warning(f"Failed to push step {step} to Hub: {e}")
