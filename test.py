def _check_torch_cuda() -> tuple[bool, str]:
    try:
        import torch
    except Exception as exc:
        return False, f"torch import failed: {exc}"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "unknown"
        return True, f"torch sees CUDA device: {name}"
    return False, "torch.cuda.is_available() is False"


def _check_cuda() -> None:
    ok, info = _check_torch_cuda()
    if ok:
        print("CUDA available")
        print(info)
    else:
        print("CUDA not available")
        print(info)


if __name__ == "__main__":
    _check_cuda()
