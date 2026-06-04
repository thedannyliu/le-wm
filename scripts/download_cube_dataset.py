from pathlib import Path

from huggingface_hub import hf_hub_download


OUT_DIR = Path("/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/downloads")
OUT_DIR.mkdir(parents=True, exist_ok=True)

path = hf_hub_download(
    repo_id="quentinll/lewm-cube",
    filename="cube_single_expert.tar.zst",
    repo_type="dataset",
    local_dir=str(OUT_DIR),
    local_dir_use_symlinks=False,
)
print(path, flush=True)
