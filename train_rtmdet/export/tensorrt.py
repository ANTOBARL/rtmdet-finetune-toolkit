# =============================================================================
# IDLE — TensorRT export is not active in the current workflow.
#
# To enable TensorRT export:
#   1. Install MMDeploy from source with TensorRT support.
#   2. Set run_export = true in iperparameter_config.txt.
#   3. Set mmdeploy_root in iperparameter_config.txt.
#   4. Ensure TENSORRT_DIR, CUDNN_DIR are set and trtexec is in PATH.
#   5. Run export_rtmdet_to_tensorrt.py from the project root.
#
# The standalone script export_rtmdet_to_tensorrt.py contains the full
# export logic and can be launched directly once the environment is ready.
# =============================================================================

def export_tensorrt(*args, **kwargs):
    raise NotImplementedError(
        "TensorRT export is currently IDLE. "
        "See export_rtmdet_to_tensorrt.py and the README for setup instructions."
    )
