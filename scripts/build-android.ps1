# Build Android debug .apk via Godot headless export.
# Pipeline: rebuild plugin AAR -> sync to client/addons/ -> Godot --export-debug Android.
# Usage from repo root: .\scripts\build-android.ps1
#
# Pattern lifted from the WaterBloom project (same toolchain layout).
$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$client = Join-Path $repo "client"
$pluginDir = Join-Path $repo "android-plugin"
$addons = Join-Path $client "addons"
$buildDir = Join-Path $repo "build"
$outputApk = Join-Path $buildDir "local-ai-rnd-debug.apk"

# Resolve toolchain paths (zip-installed JDK + Android SDK).
$env:JAVA_HOME = "C:\dev\jdk17\jdk-17.0.13+11"
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
$env:Path = "$env:JAVA_HOME\bin;$env:Path"

if (-not (Test-Path "$env:JAVA_HOME\bin\java.exe")) {
    Write-Host "[android] JDK 17 not found at $env:JAVA_HOME" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path "$env:ANDROID_HOME\platforms")) {
    Write-Host "[android] Android SDK not found at $env:ANDROID_HOME" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $buildDir)) { New-Item -ItemType Directory -Path $buildDir | Out-Null }

Write-Host "[android] Step 1/3 — Building plugin AAR" -ForegroundColor Cyan
Push-Location $pluginDir
try {
    & .\gradlew.bat ":draw_moment_plugin:assembleRelease" --console=plain --no-daemon
    if ($LASTEXITCODE -ne 0) { throw "AAR build failed (exit $LASTEXITCODE)" }
} finally {
    Pop-Location
}

$srcAar = Join-Path $pluginDir "draw_moment_plugin\build\outputs\aar\draw_moment_plugin-release.aar"
if (-not (Test-Path $srcAar)) { throw "AAR missing at $srcAar" }

# Two destinations:
# - addons root: Godot's Android template auto-includes this via
#   fileTree(dir: addons, include: '*.aar')
# - .gdap subdir: keeps the .gdap binary= entry valid
Write-Host "[android] Step 2/3 — Syncing AAR to client/addons/" -ForegroundColor Cyan
Copy-Item $srcAar (Join-Path $addons "draw_moment_plugin.aar") -Force
Copy-Item $srcAar (Join-Path $addons "draw_moment_plugin\draw_moment_plugin.aar") -Force

# Godot's Android template does fileTree(dir: addons, include: '*.aar')
# which only picks up the AARs that physically sit in addons/. Our
# plugin AAR's transitive dependency on onnxruntime-android (Phase 2)
# isn't followed, so the runtime libonnxruntime.so + Java classes
# never make it into the APK. Copy the ORT AAR straight from Gradle's
# downloaded-artifacts cache so Godot's template picks it up.
$ortAar = Get-ChildItem `
    "$env:USERPROFILE\.gradle\caches\modules-2\files-2.1\com.microsoft.onnxruntime\onnxruntime-android\1.19.2" `
    -Recurse -Filter "*.aar" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($ortAar) {
    Copy-Item $ortAar.FullName (Join-Path $addons "onnxruntime-android.aar") -Force
    $ortMb = [math]::Round($ortAar.Length / 1MB, 1)
    Write-Host "[android]   + onnxruntime-android.aar ($ortMb MB)" -ForegroundColor DarkGreen
} else {
    Write-Host "[android] ORT AAR not in gradle cache yet — Step 1 should have downloaded it" -ForegroundColor Red
    exit 1
}

if (Test-Path $outputApk) { Remove-Item $outputApk -Force }

Write-Host "[android] Step 3/3 — Exporting debug APK via Godot" -ForegroundColor Cyan
& (Join-Path $PSScriptRoot "godot.ps1") --headless --path $client --export-debug "Android"
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Host "[android] Godot export FAILED (exit $exitCode)" -ForegroundColor Red
    exit $exitCode
}

if (-not (Test-Path $outputApk)) {
    Write-Host "[android] Export reported success but .apk missing at $outputApk" -ForegroundColor Red
    exit 1
}

$size = (Get-Item $outputApk).Length / 1MB
Write-Host ("[android] OK: {0} ({1:N1} MB)" -f $outputApk, $size) -ForegroundColor Green
