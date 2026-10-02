param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
# setup.bat이 만든 .venv가 있으면 사용하고, 없으면 PATH의 python + .packages를 사용합니다.
$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
    $py = 'python'
    $env:PYTHONPATH = "$PSScriptRoot\.packages;$PSScriptRoot"
}
& $py -m uvicorn server:app --host 127.0.0.1 --port $Port --ws-max-size 6291456
