param(
    [string]$Python = "python",
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

$envOverride = $env:VENV_PATH
if ($envOverride) {
    $VenvPath = $envOverride
}

$hasNonAscii = $false
foreach ($ch in $VenvPath.ToCharArray()) {
    if ([int][char]$ch -gt 127) { $hasNonAscii = $true; break }
}
if ($hasNonAscii) {
    Write-Warning "Venv path contains non-ASCII characters; MuJoCo on Windows may fail to load assets."
}

if (-not (Test-Path $VenvPath)) {
    Write-Host "Creating venv at $VenvPath ..."
    & $Python -m venv $VenvPath
}

$activate = Join-Path $VenvPath "Scripts\\Activate.ps1"
. $activate

Write-Host "Upgrading pip ..."
python -m pip install --upgrade pip

Write-Host "Installing requirements ..."
pip install -r requirements.txt

Write-Host "Done. Activate with:`n. $activate"
