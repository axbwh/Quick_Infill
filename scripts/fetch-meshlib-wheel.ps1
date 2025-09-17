param(
    [string]$BlenderPath,
    [string]$MeshlibVersion = "",
    [string]$OutDir = "wheels"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Find-BlenderExe {
    param([string]$Hint)
    if ($Hint -and (Test-Path $Hint)) { return $Hint }
    $candidates = @(
        "$env:ProgramFiles\Blender Foundation\Blender 4.5\blender.exe",
        "$env:ProgramFiles\Blender Foundation\Blender\blender.exe",
        "$env:ProgramFiles(x86)\Steam\steamapps\common\Blender\blender.exe"
    )
    foreach ($p in $candidates) { if (Test-Path $p) { return $p } }
    throw "Could not locate blender.exe. Pass -BlenderPath 'C:\\Path\\to\\blender.exe'"
}

$blenderExe = Find-BlenderExe -Hint $BlenderPath
Write-Host "Using Blender at: $blenderExe"

# Always use a temporary Python file to avoid quoting issues with --python-expr
$tmpInfo = New-TemporaryFile
@(
    'import sys, sysconfig',
    'print("PYVER=", sys.version.split()[0])',
    'print("ABI=", sysconfig.get_config_var("SOABI") or "")',
    'print("PLAT=", sysconfig.get_platform())',
    'print("PYMAJOR=", sys.version_info.major)',
    'print("PYMINOR=", sys.version_info.minor)'
) | Set-Content -Path $tmpInfo -Encoding UTF8
try {
    $raw = & $blenderExe --background --factory-startup --python $tmpInfo
} finally {
    Remove-Item $tmpInfo -Force -ErrorAction SilentlyContinue
}

$info = $raw | Select-String -Pattern '^(PYVER|ABI|PLAT|PYMAJOR|PYMINOR)\s*='
$infoList = @($info)
if ($infoList.Count -eq 0) { throw "Failed to query Blender Python info" }

$map = @{}
foreach ($l in $infoList) {
    $line = $l.ToString()
    $parts = $line.Split('=',[System.StringSplitOptions]::RemoveEmptyEntries)
    if ($parts.Count -ge 2) {
        $key = $parts[0].Trim()
        $val = ($parts[1..($parts.Count-1)] -join '=').Trim()
        $map[$key] = $val
    }
}
$pyMajor = [int]$map['PYMAJOR']
$pyMinor = [int]$map['PYMINOR']
$tag = "cp${pyMajor}${pyMinor}-cp${pyMajor}${pyMinor}"

# Use Blender's pip download to ensure compatible wheel (via temp file for reliability)
$tmpWhich = New-TemporaryFile
@('import sys', 'print(sys.executable)') | Set-Content -Path $tmpWhich -Encoding UTF8
try {
    $pyRaw = & $blenderExe --background --factory-startup --python $tmpWhich
} finally {
    Remove-Item $tmpWhich -Force -ErrorAction SilentlyContinue
}
# Extract a sane python.exe path from Blender's output
$pyCandidates = @($pyRaw | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ -match 'python(\.exe)?$' -or $_ -match '^[A-Za-z]:\\' })
if ($pyCandidates.Count -eq 0) {
    throw "Failed to resolve Blender's Python path. Output was:`n$($pyRaw -join [Environment]::NewLine)"
}
$py = $pyCandidates[-1].Trim('"')
Write-Host "Blender Python: $py"
& "$py" -m ensurepip --upgrade
& "$py" -m pip install --upgrade pip

$pkg = if ($MeshlibVersion) { "meshlib==$MeshlibVersion" } else { "meshlib" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Write-Host "Downloading $pkg wheel compatible with Blender into $OutDir ..."
& "$py" -m pip download $pkg --only-binary=:all: --dest $OutDir

# Pick the wheel matching cp tag
$wheels = Get-ChildItem $OutDir -Filter "*.whl"
if (-not $wheels) { throw "No wheels downloaded" }
$selected = $wheels | Where-Object { $_.Name -match "$tag" } | Select-Object -First 1
if (-not $selected) { $selected = $wheels | Select-Object -First 1 }

Write-Host "Selected wheel: $($selected.Name)"

# Update blender_manifest.toml wheels list
$manifest = Get-Content "blender_manifest.toml" -Raw
$relPath = "./$($OutDir)/$($selected.Name)"

if ($manifest -notmatch "(?ms)^wheels\s*=\s*\[") {
    # Insert wheels = [ ... ] block after license section
    $manifest = $manifest -replace '(?ms)(license\s*=\s*\[[^\]]*\]\s*)', "`$1`r`n`r`nwheels = [`r`n  '$relPath'`r`n]`r`n"
} else {
    # Replace existing list contents
    $manifest = $manifest -replace '(?ms)^wheels\s*=\s*\[[^\]]*\]', "wheels = [`r`n  '$relPath'`r`n]"
}

Set-Content -Path "blender_manifest.toml" -Value $manifest -Encoding UTF8
Write-Host "Updated blender_manifest.toml with wheels entry: $relPath"
