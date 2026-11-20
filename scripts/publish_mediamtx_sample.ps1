param(
    [Parameter(Mandatory = $true)][string]$VideoPath,
    [string]$CameraPath = 'CAM_TEST'
)
$ErrorActionPreference = 'Stop'

if ($CameraPath -notmatch '^[A-Za-z0-9_/-]{1,120}$') {
    throw 'CameraPath may contain only letters, numbers, underscore, hyphen, and slash.'
}
$source = (Resolve-Path -LiteralPath $VideoPath).Path
$ffmpeg = Get-Command ffmpeg -ErrorAction Stop
$url = "rtsp://127.0.0.1:8554/$CameraPath"
Write-Host "Publishing $source to $url. Press Ctrl+C to stop."
& $ffmpeg.Source -re -stream_loop -1 -i $source -map '0:v:0' -c:v copy -an -f rtsp $url
if ($LASTEXITCODE -ne 0) {
    throw "FFmpeg exited with code $LASTEXITCODE. Check the source codec and MediaMTX listener."
}
