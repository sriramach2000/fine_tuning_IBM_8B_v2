#!/bin/bash
# ============================================================================
# Launch EC2 instance to export dataset archive and upload to S3
#
# The instance auto-terminates after the export completes (or fails).
# All data stays within us-east-1 (free S3 transfer).
#
# Prerequisites:
#   - AWS CLI configured with permissions to launch EC2, create IAM roles
#   - The SageMaker role (granite-8b-avb-tsn-finetuning-sagemaker-role) must
#     have read access to both source buckets + write access to the destination
#
# Usage:
#   bash scripts/launch_ec2_export.sh
#   bash scripts/launch_ec2_export.sh s3://my-bucket/archives/
#
# Author: Sriram Acharya
# ============================================================================

set -euo pipefail

# ---- Configuration ----
REGION="us-east-1"
INSTANCE_TYPE="t3.medium"        # 2 vCPU, 4GB RAM — plenty for downloading
VOLUME_SIZE_GB=30                # ~10GB raw + ~1.5GB splits + archive + headroom
AMI_ID="ami-0c7217cdde317cfec"   # Amazon Linux 2023 (us-east-1) — update if needed
KEY_NAME=""                      # Set to your key pair name if you want SSH access
IAM_ROLE="granite-8b-avb-tsn-finetuning-sagemaker-role"

# Destination for the final archive
UPLOAD_S3_URI="${1:-s3://granite-8b-training-outputs/archives/}"

# Git repo (the export script + prepare_automotive_data.py are needed)
REPO_URL="https://github.com/YOUR_USERNAME/fine_tuning_IBM_8B_v2.git"
REPO_BRANCH="main"

echo "============================================="
echo "EC2 Dataset Export Launcher"
echo "============================================="
echo "  Region:         ${REGION}"
echo "  Instance type:  ${INSTANCE_TYPE}"
echo "  Volume:         ${VOLUME_SIZE_GB} GB"
echo "  Upload to:      ${UPLOAD_S3_URI}"
echo "============================================="

# ---- User data script (runs on instance boot) ----
# This script:
#   1. Installs Python, pip, git, pigz
#   2. Clones the repo
#   3. Installs Python dependencies
#   4. Runs the export with --upload-to-s3
#   5. Terminates the instance (self-destruct)

USERDATA=$(cat <<'USERDATA_EOF'
#!/bin/bash
set -euxo pipefail

# Log everything
exec > /var/log/export-dataset.log 2>&1
echo "=== Dataset export started at $(date -u) ==="

# Get instance ID for self-termination
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)

# Cleanup function — ALWAYS terminate, even on failure
cleanup() {
    echo "=== Export finished at $(date -u) ==="
    echo "=== Terminating instance ${INSTANCE_ID} ==="
    aws ec2 terminate-instances --instance-ids "${INSTANCE_ID}" --region "${REGION}" || true
}
trap cleanup EXIT

# Install dependencies
dnf install -y python3.11 python3.11-pip git pigz

# Clone repo
cd /tmp
git clone --depth 1 -b __REPO_BRANCH__ __REPO_URL__ export_repo
cd export_repo

# Install Python deps
python3.11 -m pip install boto3 pyyaml tqdm python-dotenv

# Run the export
python3.11 scripts/export_dataset_archive.py \
    --upload-to-s3 __UPLOAD_S3_URI__ \
    --output-path /tmp/granite-8b-automotive-dataset.tar.gz \
    --workers 32

echo "=== Export completed successfully ==="
USERDATA_EOF
)

# Substitute variables into userdata
USERDATA="${USERDATA//__REPO_URL__/$REPO_URL}"
USERDATA="${USERDATA//__REPO_BRANCH__/$REPO_BRANCH}"
USERDATA="${USERDATA//__UPLOAD_S3_URI__/$UPLOAD_S3_URI}"

# Base64 encode for API
USERDATA_B64=$(echo "$USERDATA" | base64)

# ---- Check for IAM instance profile ----
echo "[Launcher] Checking IAM instance profile..."
PROFILE_NAME="${IAM_ROLE}"
if ! aws iam get-instance-profile --instance-profile-name "${PROFILE_NAME}" --region "${REGION}" >/dev/null 2>&1; then
    echo "[Launcher] Creating instance profile ${PROFILE_NAME}..."
    aws iam create-instance-profile --instance-profile-name "${PROFILE_NAME}" --region "${REGION}"
    aws iam add-role-to-instance-profile --instance-profile-name "${PROFILE_NAME}" --role-name "${IAM_ROLE}" --region "${REGION}"
    echo "[Launcher] Waiting for profile propagation..."
    sleep 15
fi

# ---- Launch EC2 instance ----
echo "[Launcher] Launching EC2 instance..."

LAUNCH_ARGS=(
    ec2 run-instances
    --region "${REGION}"
    --instance-type "${INSTANCE_TYPE}"
    --image-id "${AMI_ID}"
    --iam-instance-profile "Name=${PROFILE_NAME}"
    --block-device-mappings "DeviceName=/dev/xvda,Ebs={VolumeSize=${VOLUME_SIZE_GB},VolumeType=gp3}"
    --user-data "${USERDATA_B64}"
    --instance-initiated-shutdown-behavior terminate
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=granite-dataset-export},{Key=Purpose,Value=dataset-archive-export},{Key=AutoTerminate,Value=true}]"
    --count 1
)

# Add key pair if specified
if [[ -n "${KEY_NAME}" ]]; then
    LAUNCH_ARGS+=(--key-name "${KEY_NAME}")
fi

RESULT=$(aws "${LAUNCH_ARGS[@]}" --output json)
INSTANCE_ID=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['Instances'][0]['InstanceId'])")

echo ""
echo "============================================="
echo "[Launcher] Instance launched: ${INSTANCE_ID}"
echo "============================================="
echo "  The instance will:"
echo "    1. Install dependencies"
echo "    2. Clone the repo"
echo "    3. Run the export pipeline (~5 min)"
echo "    4. Upload archive to ${UPLOAD_S3_URI}"
echo "    5. AUTO-TERMINATE itself"
echo ""
echo "  Monitor progress:"
echo "    aws ec2 describe-instances --instance-ids ${INSTANCE_ID} --query 'Reservations[].Instances[].State.Name' --output text"
echo ""
echo "  View logs (if SSH key configured):"
echo "    ssh ec2-user@<public-ip> 'cat /var/log/export-dataset.log'"
echo ""
echo "  Download archive when done:"
echo "    aws s3 cp ${UPLOAD_S3_URI}granite-8b-automotive-dataset.tar.gz ."
echo "============================================="
