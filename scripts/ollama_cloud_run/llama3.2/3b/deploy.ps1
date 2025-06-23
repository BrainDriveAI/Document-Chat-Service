<#
.SYNOPSIS
Deploys a Google Cloud Run service with specified configurations.

.DESCRIPTION
This script deploys a Google Cloud Run service using the gcloud CLI.
It allows for configuration of CPU, memory, execution environment, and
unauthenticated access.

.PARAMETER ServiceName
The name of the Google Cloud Run service to deploy.
If not provided, the script will attempt to deploy without a specific service name,
relying on gcloud's default behavior or context.

.PARAMETER Region
The Google Cloud region where the service will be deployed.
If not provided, gcloud will use the configured default region.

.EXAMPLE
.\deploy.ps1 -ServiceName "my-service" -Region "us-central1"

.EXAMPLE
.\deploy.ps1 # Deploys with default gcloud settings and no specific service name

.NOTES
Ensure you are authenticated to gcloud CLI and have the necessary
permissions to deploy Cloud Run services.
#>
param(
    [string]$ServiceName = "ollama-llama3-2-3b"
)

Write-Host "Starting Google Cloud Run deployment..." -ForegroundColor Green

# Define common deployment arguments
$gcloudArgs = @(
    "--source=."
    "--cpu=4"
    "--memory=8Gi"
    "--set-env-vars=OLLAMA_NUM_PARALLEL=4"
    "--execution-environment=gen2"
    "--allow-unauthenticated"
    "--no-cpu-throttling"
    "--max-instances=10"
    "--region=us-central1"
)

# Add ServiceName if provided
if (-not [string]::IsNullOrEmpty($ServiceName)) {
    $gcloudArgs += $ServiceName
    Write-Host "Deploying service: $ServiceName" -ForegroundColor Cyan
} else {
    Write-Host "No specific service name provided. gcloud will use default context or inferred name." -ForegroundColor Yellow
}

# Execute the gcloud command
try {
    Write-Host "Executing command: gcloud run deploy $($gcloudArgs -join ' ')" -ForegroundColor DarkGray
    gcloud alpha run deploy @gcloudArgs

    if ($LASTEXITCODE -eq 0) {
        Write-Host "Google Cloud Run deployment successful!" -ForegroundColor Green
    } else {
        Write-Error "Google Cloud Run deployment failed with exit code: $LASTEXITCODE"
        exit $LASTEXITCODE # Exit with the gcloud error code
    }
}
catch {
    Write-Error "An error occurred during gcloud execution: $($_.Exception.Message)"
    exit 1 # Exit with a generic error code
}