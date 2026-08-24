# Activate the shared GenAI venv and set the environment variables every project
# expects. Run from anywhere:   .\activate.ps1
#
# The venv lives OUTSIDE OneDrive on purpose -- a ~9 GB virtualenv inside a
# synced folder churns the sync client and can corrupt hardlinked packages.

$VenvPath = Join-Path $HOME ".venvs\genai"
$Activate = Join-Path $VenvPath "Scripts\Activate.ps1"

if (-not (Test-Path $Activate)) {
    Write-Host "No venv at $VenvPath" -ForegroundColor Red
    Write-Host "Create it with:" -ForegroundColor Yellow
    Write-Host "  uv venv --python 3.12 `"$VenvPath`"" -ForegroundColor Yellow
    Write-Host "See SETUP.md for the full one-time setup." -ForegroundColor Yellow
    exit 1
}

& $Activate

# --- quiet the noise -------------------------------------------------------
# Windows without Developer Mode can't symlink, so the HF cache warns on every
# model load. Harmless, and very loud.
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:TOKENIZERS_PARALLELISM          = "false"

# uv hardlinks from its cache by default; OneDrive's filesystem rejects that
# (os error 396). Copy mode is slower to install and actually works.
$env:UV_LINK_MODE = "copy"

$py = Join-Path $VenvPath "Scripts\python.exe"
$ver = & $py -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"

Write-Host ""
Write-Host "  GenAI workspace" -ForegroundColor Cyan
Write-Host "  venv    $VenvPath"
Write-Host "  python  $ver"

$gpu = & $py -c @"
try:
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        print(f'{p.name}  |  {free/1024**3:.1f} of {total/1024**3:.1f} GB free  |  '
              f'torch {torch.__version__}')
    else:
        print('CUDA NOT AVAILABLE - see SETUP.md')
except ImportError:
    print('torch not installed - see SETUP.md')
"@
Write-Host "  gpu     $gpu"
Write-Host ""
Write-Host "  Projects:  01_rag_local  02_lora_text  03_lora_image" -ForegroundColor DarkGray
Write-Host "             04_lora_voice  05_mcp_server  06_local_gpu_inference" -ForegroundColor DarkGray
Write-Host "  Check:     python 00_shared\check_env.py" -ForegroundColor DarkGray
Write-Host ""
