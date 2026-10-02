# K-Trip 첫 설치: .venv 생성 + 패키지 설치 + cloudflared 다운로드 + 가중치 확인.
# 여러 번 실행해도 안전합니다. 이미 된 단계는 건너뜁니다.
param([switch]$Recreate)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$venvPy = Join-Path $root '.venv\Scripts\python.exe'

function Step($msg) { Write-Host ""; Write-Host "== $msg" -ForegroundColor Cyan }
function Fail($msg) {
    Write-Host ""
    Write-Host "[!] $msg" -ForegroundColor Red
    Write-Host ""
    exit 1
}

# ---- 1. Python 3.10+ 찾기 -------------------------------------------------------------
# PyTorch 휠이 안정적인 3.12 → 3.11 → 3.10 순으로 먼저 찾고, 없으면 PATH의 python.
function Find-Python {
    $candidates = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in '3.12', '3.11', '3.10', '3.13') { $candidates += , @('py', "-$v") }
    }
    $candidates += , @('python')
    foreach ($c in $candidates) {
        $exe = $c[0]; $pre = @($c | Select-Object -Skip 1)
        try {
            $out = & $exe @pre -c "import sys; print('%d.%d' % sys.version_info[:2]); print(sys.executable)" 2>$null
        } catch { continue }
        if ($LASTEXITCODE -ne 0 -or -not $out) { continue }
        $ver = [version]$out[0]
        if ($ver -ge [version]'3.10') { return @{ Exe = $out[1]; Ver = $out[0] } }
    }
    return $null
}

Step "1/5 Python 확인"
if ($Recreate -and (Test-Path '.venv')) {
    Write-Host "기존 .venv 삭제 (-Recreate)"
    Remove-Item -Recurse -Force '.venv'
}
if (Test-Path $venvPy) {
    Write-Host "기존 가상환경 사용: $venvPy"
} else {
    $py = Find-Python
    if (-not $py) {
        Fail ("Python 3.10 이상을 찾지 못했습니다.`n" +
              "    https://www.python.org/downloads/ 에서 Python 3.12를 설치하세요.`n" +
              "    설치 화면에서 'Add python.exe to PATH'를 체크한 뒤 setup.bat을 다시 실행하세요.")
    }
    Write-Host "Python $($py.Ver): $($py.Exe)"

    Step "2/5 가상환경(.venv) 생성"
    & $py.Exe -m venv .venv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPy)) { Fail ".venv 생성에 실패했습니다." }
    Write-Host "생성 완료: $root\.venv"
}

# ---- 2. 패키지 설치 --------------------------------------------------------------------
Step "3/5 패키지 설치 (처음에는 PyTorch 때문에 몇 분 걸립니다)"
& $venvPy -m pip install --upgrade pip --disable-pip-version-check -q
if ($LASTEXITCODE -ne 0) { Fail "pip 업그레이드 실패. 인터넷 연결을 확인하세요." }
& $venvPy -m pip install -r requirements.txt --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { Fail "패키지 설치 실패. 위 오류 메시지를 확인하세요." }
& $venvPy -c "import torch, torchvision, fastapi, uvicorn, qrcode, cv2; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { Fail "설치된 패키지를 불러오지 못했습니다." }

# ---- 3. cloudflared ---------------------------------------------------------------------
# 버전을 고정하고 릴리스에 공개된 SHA256으로 검증합니다.
Step "4/5 cloudflared (공개 https 링크용)"
$cfVer = '2026.9.1'
$cfSha = '2837888cc0f5d58f15b6dc478376de90b4d3ba5241c7947455d1e0a0df429712'
$cfUrl = "https://github.com/cloudflare/cloudflared/releases/download/$cfVer/cloudflared-windows-amd64.exe"
$cf = Join-Path $PSScriptRoot 'cloudflared.exe'

if (Test-Path $cf) {
    Write-Host "이미 있음: $cf"
} elseif (Get-Command cloudflared -ErrorAction SilentlyContinue) {
    Write-Host "PATH에 있는 cloudflared 사용: $((Get-Command cloudflared).Source)"
} else {
    Write-Host "다운로드: $cfUrl"
    Write-Host "저장 위치: $cf  (약 30 MB)"
    $tmp = "$cf.download"
    try {
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $cfUrl -OutFile $tmp -UseBasicParsing
    } catch {
        Remove-Item $tmp -ErrorAction SilentlyContinue
        Fail ("cloudflared 다운로드 실패: $($_.Exception.Message)`n" +
              "    직접 받으려면 https://github.com/cloudflare/cloudflared/releases 에서`n" +
              "    cloudflared-windows-amd64.exe를 받아 이름을 cloudflared.exe로 바꾸고 $PSScriptRoot 에 넣으세요.")
    }
    $hash = (Get-FileHash -Algorithm SHA256 $tmp).Hash.ToLower()
    if ($hash -ne $cfSha) {
        Remove-Item $tmp -ErrorAction SilentlyContinue
        Fail "다운로드한 파일의 SHA256이 공개값과 다릅니다. 파일을 삭제했습니다.`n    받은 값: $hash`n    기대 값: $cfSha"
    }
    Move-Item $tmp $cf
    Write-Host "SHA256 검증 완료"
}

# ---- 4. 모델 가중치 ---------------------------------------------------------------------
Step "5/5 모델 가중치 확인"
$meta = Get-Content (Join-Path $root 'checkpoints\meta.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$weights = Join-Path $root "checkpoints\$($meta.weights)"
$weightsOk = Test-Path $weights
if ($weightsOk) {
    Write-Host ("있음: {0} ({1:N1} MB)" -f $weights, ((Get-Item $weights).Length / 1MB))
} else {
    Write-Host "[!] 가중치 파일이 없습니다." -ForegroundColor Yellow
    Write-Host "    가지고 있는 $($meta.weights) 파일을 아래 위치에 복사하세요:"
    Write-Host "        $weights"
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
if ($weightsOk) {
    Write-Host "  설치 완료! 이제 share.bat을 실행하면 공개 링크와 QR이 나옵니다." -ForegroundColor Green
} else {
    Write-Host "  설치 완료. 가중치 파일을 넣은 뒤 share.bat을 실행하세요." -ForegroundColor Yellow
}
Write-Host "============================================================" -ForegroundColor Green
