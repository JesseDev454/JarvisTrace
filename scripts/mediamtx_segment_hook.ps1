param()
$ErrorActionPreference = 'Stop'

$token = $env:JARVISTRACE_MEDIAMTX_TOKEN
if ([string]::IsNullOrWhiteSpace($token)) {
    throw 'JARVISTRACE_MEDIAMTX_TOKEN must be set in the MediaMTX process environment.'
}
if ([string]::IsNullOrWhiteSpace($env:MTX_PATH) -or [string]::IsNullOrWhiteSpace($env:MTX_SEGMENT_PATH)) {
    throw 'MediaMTX did not provide the completed stream and segment paths.'
}

$payload = @{
    path = $env:MTX_PATH
    segment_path = $env:MTX_SEGMENT_PATH
} | ConvertTo-Json -Compress

Invoke-RestMethod `
    -Method Post `
    -Uri 'http://127.0.0.1:8770/api/mediamtx/segment' `
    -Headers @{ Authorization = "Bearer $token" } `
    -ContentType 'application/json' `
    -Body $payload | Out-Null
