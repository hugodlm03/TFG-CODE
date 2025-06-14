# start_clients.ps1
$csvFiles = Get-ChildItem .\nodos\*.csv
foreach ($file in $csvFiles) {
  $name = $file.BaseName -replace '_', ' '
  Start-Process python -ArgumentList @(
    "-m", "src.flower_client_tree",
    "--csv", "`"$($file.FullName)`"",
    "--name", "`"$name`"",
    "--server", "127.0.0.1:8085"
  )
}
