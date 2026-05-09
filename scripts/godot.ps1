# Godot CLI helper — invokes the winget-installed Godot console binary so stdout/stderr stream.
# Usage from repo root: pwsh scripts\godot.ps1 --version
#
# Pattern lifted from the WaterBloom project (same dev box, same Godot install).
$ErrorActionPreference = "Stop"

$candidates = @(
    "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64_console.exe"
)

# Fallback: glob any installed Godot console exe.
if (-not (Test-Path $candidates[0])) {
    $found = Get-ChildItem -Path "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter "Godot_*_console.exe" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($found) { $candidates = @($found.FullName) }
}

$godot = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $godot) {
    Write-Error "Godot binary not found. Run: winget install GodotEngine.GodotEngine"
    exit 1
}

& $godot @args
exit $LASTEXITCODE
