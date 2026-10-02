# 서버 기동 → 모델 로드 대기 → Cloudflare 임시 터널 → 공개 링크 + QR 출력.
# 이 창을 닫거나 Ctrl+C를 누르면 서버와 링크가 함께 종료됩니다.
param(
    [int]$Port = 8765,
    [string]$Name = $env:COMPUTERNAME,
    [switch]$NoQrWindow
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'

function Fail($msg) {
    Write-Host ""
    Write-Host "[!] $msg" -ForegroundColor Red
    Write-Host ""
    exit 1
}
function Test-Port([int]$p) {
    $c = New-Object Net.Sockets.TcpClient
    try { $c.Connect('127.0.0.1', $p); $true } catch { $false } finally { $c.Dispose() }
}
function Show-Tail($path) {
    if (Test-Path $path) { Get-Content $path -Tail 20 | ForEach-Object { Write-Host "    $_" } }
}

# ---- 준비물 확인 ------------------------------------------------------------------------
$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { Fail "가상환경(.venv)이 없습니다. setup.bat을 먼저 실행하세요." }

$meta = Get-Content (Join-Path $root 'checkpoints\meta.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$weights = Join-Path $root "checkpoints\$($meta.weights)"
if (-not $env:KTRIP_CHECKPOINT -and -not (Test-Path $weights)) {
    Fail "모델 가중치가 없습니다. $($meta.weights) 파일을 아래 위치에 복사하세요:`n        $weights"
}

$cf = Join-Path $PSScriptRoot 'cloudflared.exe'
if (-not (Test-Path $cf)) {
    $onPath = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($onPath) { $cf = $onPath.Source } else { Fail "cloudflared.exe가 없습니다. setup.bat을 먼저 실행하세요." }
}

# 포트가 사용 중이면(예: run.bat이 켜져 있음) 다음 빈 포트를 씁니다.
$start = $Port
while (Test-Port $Port) {
    $Port++
    if ($Port -gt $start + 20) { Fail "$start~$Port 포트가 모두 사용 중입니다." }
}
if ($Port -ne $start) { Write-Host "포트 $start 은(는) 사용 중이라 $Port 을(를) 사용합니다." -ForegroundColor Yellow }

$logs = Join-Path $root 'logs'
New-Item -ItemType Directory -Force $logs | Out-Null
$srvOut = Join-Path $logs 'server.out.log'; $srvErr = Join-Path $logs 'server.err.log'
$cfOut = Join-Path $logs 'tunnel.out.log'; $cfErr = Join-Path $logs 'tunnel.err.log'
Remove-Item $srvOut, $srvErr, $cfOut, $cfErr -ErrorAction SilentlyContinue

$srv = $null; $tun = $null
try {
    # ---- 1. 서버 ------------------------------------------------------------------------
    Write-Host ""
    Write-Host "[1/3] 서버 시작 (http://localhost:$Port) - 모델 로드까지 기다립니다 ..." -ForegroundColor Cyan
    $srv = Start-Process -FilePath $py -WorkingDirectory $root -NoNewWindow -PassThru `
        -ArgumentList '-m', 'uvicorn', 'server:app', '--host', '127.0.0.1', '--port', "$Port", '--ws-max-size', '6291456' `
        -RedirectStandardOutput $srvOut -RedirectStandardError $srvErr

    $ready = $false
    foreach ($i in 1..300) {
        Start-Sleep -Seconds 1
        if ($srv.HasExited) { break }
        try {
            $h = Invoke-RestMethod "http://127.0.0.1:$Port/api/health" -TimeoutSec 2
            if ($h.ready) { $ready = $true; break }
        } catch { }
    }
    if (-not $ready) {
        Write-Host "[!] 서버가 준비되지 않았습니다. 서버 로그 마지막 부분:" -ForegroundColor Red
        Show-Tail $srvErr
        Fail "전체 로그: $srvErr"
    }
    Write-Host "      준비 완료: $($h.model.arch) / device $($h.model.device)"

    # ---- 2. 터널 ------------------------------------------------------------------------
    Write-Host "[2/3] 공개 https 링크 생성 중 ..." -ForegroundColor Cyan
    $tun = Start-Process -FilePath $cf -NoNewWindow -PassThru `
        -ArgumentList 'tunnel', '--no-autoupdate', '--url', "http://127.0.0.1:$Port" `
        -RedirectStandardOutput $cfOut -RedirectStandardError $cfErr

    $url = $null
    foreach ($i in 1..60) {
        Start-Sleep -Seconds 1
        if ($tun.HasExited) { break }
        $text = (Get-Content $cfOut, $cfErr -Raw -ErrorAction SilentlyContinue) -join "`n"
        $m = [regex]::Match($text, 'https://[a-z0-9-]+\.trycloudflare\.com')
        if ($m.Success) { $url = $m.Value; break }
    }
    if (-not $url) {
        Write-Host "[!] 60초 안에 링크를 받지 못했습니다. 네트워크(방화벽/프록시)가 막고 있을 수 있습니다." -ForegroundColor Red
        Show-Tail $cfErr
        Write-Host ""
        Write-Host "    이 PC에서만 시연하려면 run.bat 후 http://localhost:$start 를 여세요."
        Fail "전체 로그: $cfErr"
    }

    # ---- 3. 링크 + QR ---------------------------------------------------------------------
    Write-Host "[3/3] 외부 접속 확인 중 ..." -ForegroundColor Cyan
    $reachable = $false
    foreach ($i in 1..20) {
        try { if ((Invoke-RestMethod "$url/api/health" -TimeoutSec 5).ready) { $reachable = $true; break } } catch { }
        Start-Sleep -Seconds 2
    }

    $png = Join-Path $root 'share_qr.png'
    Set-Content -Path (Join-Path $root 'share_link.txt') -Value $url -Encoding ASCII
    Write-Host ""
    & $py (Join-Path $PSScriptRoot 'qr.py') $url $png "K-Trip  ·  $Name"
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "  [$Name] 체험 링크:" -ForegroundColor Green
    Write-Host ""
    Write-Host "      $url" -ForegroundColor White
    Write-Host ""
    if ($reachable) {
        Write-Host "  외부 접속 확인 완료. QR을 찍거나 링크를 공유하세요." -ForegroundColor Green
    } else {
        Write-Host "  아직 외부에서 확인되지 않았습니다. 30초쯤 뒤 폰으로 열어보세요." -ForegroundColor Yellow
    }
    Write-Host "  QR 이미지: $png"
    Write-Host ""
    Write-Host "  이 창을 닫거나 Ctrl+C를 누르면 링크가 사라집니다."
    Write-Host "  링크는 실행할 때마다 새로 바뀝니다."
    Write-Host "============================================================" -ForegroundColor Green
    if (-not $NoQrWindow) { Start-Process $png }

    while (-not $srv.HasExited -and -not $tun.HasExited) { Start-Sleep -Seconds 2 }
    if ($srv.HasExited) { Write-Host "[!] 서버가 종료되었습니다:" -ForegroundColor Red; Show-Tail $srvErr }
    if ($tun.HasExited) { Write-Host "[!] 터널이 끊어졌습니다:" -ForegroundColor Red; Show-Tail $cfErr }
}
finally {
    foreach ($p in $tun, $srv) {
        if ($p -and -not $p.HasExited) { try { $p.Kill() } catch { } }
    }
    Write-Host ""
    Write-Host "서버와 링크를 종료했습니다. 다시 열려면 share.bat을 실행하세요 (새 링크가 발급됩니다)."
}
