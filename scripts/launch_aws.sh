#!/bin/bash
# Launch AWS instance for Fashion ML training
set -e

REGION="us-east-1"
KEY_NAME="fashion-ml-key"
KEY_FILE="$HOME/.ssh/${KEY_NAME}.pem"
INSTANCE_TYPE="g4dn.xlarge"  # $0.526/hr, 16GB RAM, T4 GPU
VOLUME_SIZE=100  # GB

echo "=== Launching Fashion ML Instance in $REGION ==="

# Create key pair if it doesn't exist
if [ ! -f "$KEY_FILE" ]; then
  echo "Creating new SSH key pair..."
  aws ec2 delete-key-pair --region $REGION --key-name $KEY_NAME 2>/dev/null || true
  aws ec2 create-key-pair --region $REGION \
    --key-name $KEY_NAME \
    --query 'KeyMaterial' --output text > "$KEY_FILE"
  chmod 400 "$KEY_FILE"
  echo "Created key: $KEY_FILE"
else
  echo "Using existing key: $KEY_FILE"
fi

# Create security group (if it doesn't exist)
SG_NAME="fashion-ml-sg"
SG_ID=$(aws ec2 describe-security-groups --region $REGION \
  --filters Name=group-name,Values=$SG_NAME \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null)

if [ "$SG_ID" == "None" ] || [ -z "$SG_ID" ]; then
  echo "Creating security group..."
  SG_ID=$(aws ec2 create-security-group --region $REGION \
    --group-name $SG_NAME \
    --description "Fashion ML training instance" \
    --query 'GroupId' --output text)

  # Allow SSH from anywhere (you can restrict this to your IP)
  aws ec2 authorize-security-group-ingress --region $REGION \
    --group-id $SG_ID \
    --protocol tcp --port 22 --cidr 0.0.0.0/0

  echo "Created security group: $SG_ID"
else
  echo "Using existing security group: $SG_ID"
fi

# Get latest Deep Learning AMI
echo "Finding latest PyTorch Deep Learning AMI..."
AMI_ID=$(aws ec2 describe-images --region $REGION \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning AMI (Amazon Linux 2) Version *" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' --output text)

echo "Using AMI: $AMI_ID"

# Launch instance
echo "Launching $INSTANCE_TYPE instance..."
INSTANCE_ID=$(aws ec2 run-instances --region $REGION \
  --image-id $AMI_ID \
  --instance-type $INSTANCE_TYPE \
  --key-name $KEY_NAME \
  --security-group-ids $SG_ID \
  --block-device-mappings "[{\"DeviceName\":\"/dev/xvda\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE,\"VolumeType\":\"gp3\"}}]" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=fashion-ml}]" \
  --query 'Instances[0].InstanceId' --output text)

echo "Instance launched: $INSTANCE_ID"
echo "Waiting for instance to be running..."

aws ec2 wait instance-running --region $REGION --instance-ids $INSTANCE_ID

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances --region $REGION \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo ""
echo "=== Instance Ready ==="
echo "Instance ID: $INSTANCE_ID"
echo "Public IP:   $PUBLIC_IP"
echo ""
echo "Connect with:"
echo "  ssh -i $KEY_FILE ec2-user@$PUBLIC_IP"
echo ""
echo "Once connected, activate PyTorch:"
echo "  source activate pytorch"
echo ""
echo "To stop the instance (saves money):"
echo "  aws ec2 stop-instances --region $REGION --instance-ids $INSTANCE_ID"
echo ""
echo "To terminate (delete) the instance:"
echo "  aws ec2 terminate-instances --region $REGION --instance-ids $INSTANCE_ID"
