$SERVER_IP = "FILL_SERVER_IP"
$USERNAME = "root"
$SSH_PORT = "22"
$LOCAL_DATASET_ZIP = "D:\pycharm\watermark_exps\dataset.zip"
$REMOTE_DIR = "/data/watermark_exps"

if (!(Test-Path $LOCAL_DATASET_ZIP)) {
    throw "Missing dataset zip: $LOCAL_DATASET_ZIP"
}

scp -P $SSH_PORT $LOCAL_DATASET_ZIP "${USERNAME}@${SERVER_IP}:$REMOTE_DIR/"
