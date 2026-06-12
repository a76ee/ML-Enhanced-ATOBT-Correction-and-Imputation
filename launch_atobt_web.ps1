param(
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"

if ($CheckOnly) {
    Write-Host "ATOBT web launcher script is valid."
    exit 0
}

$Root = $PSScriptRoot
$AppPath = Join-Path $Root "web_app\app.py"
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

function Test-AppReady($Port) {
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/status" -TimeoutSec 2
        return ([bool]$response.ready -and $response.training.target -eq "实际移交机坪管制时间")
    } catch {
        return $false
    }
}

function Test-PortFree($Port) {
    $listener = $null
    try {
        $address = [System.Net.IPAddress]::Parse("127.0.0.1")
        $listener = New-Object System.Net.Sockets.TcpListener($address, $Port)
        $listener.Start()
        return $true
    } catch {
        return $false
    } finally {
        if ($listener -ne $null) {
            $listener.Stop()
        }
    }
}

function Get-AvailablePort($StartPort) {
    for ($port = $StartPort; $port -lt ($StartPort + 30); $port++) {
        if (Test-AppReady $port) {
            return @{ Port = $port; AlreadyRunning = $true }
        }
        if (Test-PortFree $port) {
            return @{ Port = $port; AlreadyRunning = $false }
        }
    }
    throw "No available local port found near $StartPort."
}

if (-not (Test-Path $AppPath)) {
    throw "Cannot find web app: $AppPath"
}

$portInfo = Get-AvailablePort 8765
$Port = [int]$portInfo.Port
$Url = "http://127.0.0.1:$Port/"

if (-not $portInfo.AlreadyRunning) {
    $serverCommand = @"
`$Host.UI.RawUI.WindowTitle = "ATOBT Web App Server"
Write-Host "Starting ATOBT web server on $Url"
Write-Host "Keep this window open while using the web page."
& "$Python" "$AppPath" --host "127.0.0.1" --port $Port
Write-Host ""
Read-Host "Server stopped. Press Enter to close this window"
"@
    $encoded = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($serverCommand))
    Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-NoExit",
        "-EncodedCommand",
        $encoded
    )
}

Write-Host "Waiting for ATOBT web app..."
$ready = $false
for ($i = 0; $i -lt 180; $i++) {
    Start-Sleep -Seconds 1
    if (Test-AppReady $Port) {
        $ready = $true
        break
    }
}

if (-not $ready) {
    Write-Host "The web app did not become ready in time."
    Write-Host "Please check the ATOBT Web App Server window."
    Read-Host "Press Enter to close"
    exit 1
}

Write-Host "Opening $Url"
Start-Process $Url
