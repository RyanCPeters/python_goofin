from pathlib import Path
import json
from src.utils.custom_logger import get_logger

root_info_logger = get_logger(__name__,"project info",level="INFO").warning
root_error_logger = get_logger(__name__,"project error",level="ERROR").error
code_root = Path(__file__).parent
cache_path = code_root.parent.joinpath("cache").resolve()
cache_path.mkdir(parents=True,exist_ok=True)
cfg_path = code_root.joinpath("cfgs")
caching_paths:dict = json.loads(cfg_path.joinpath("caching_paths.json").read_text())
class MUTABLE_GLOBALS:
    inpection_id_num = 0
    input_root= None
    KERNEL_HW = 3  # defined as a global constant for use in
    FMULT = 0
    TPB_CONV = 2, 2
    TPB_VECT_FORWARD = 4, 1
    INPUT_EDGE_PADDING = 1
    SAMPLES_PER_BATCH = 1
    POOL_STRIDES = 1
    output_root = code_root