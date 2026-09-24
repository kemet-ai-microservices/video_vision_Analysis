# XCLIP and MiniCPM GPU services

These services run on the NVIDIA GPU machine. The Azure-side pipeline calls
them through HTTP, normally through reverse SSH tunnels on ports 8200 and 8300.

## Architecture

- `xclip-service` keeps XCLIP loaded on the GPU. The Azure client samples eight
  video frames locally and uploads only those JPEGs to `POST /v1/classify`.
- `minicpm-caption-service` keeps MiniCPM-V loaded on the GPU and accepts one
  keyframe at `POST /v1/caption`.
- Both containers publish only on host loopback. They are not exposed to the
  public network.

## Start on the GPU machine

The machine needs Docker, the NVIDIA driver, and NVIDIA Container Toolkit.
Confirm Docker GPU access first:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

Create the environment file and set the two existing model snapshot paths:

```bash
cd gpu_services
cp .env.example .env
```

If the models should share GPU 0, leave `XCLIP_GPU=0` and `CAPTION_GPU=0`.
When VRAM is insufficient, put them on different devices, for example
`XCLIP_GPU=0` and `CAPTION_GPU=1`. Remember that ASR and OCR also consume VRAM.

Build and start:

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f xclip caption
```

Model startup may take several minutes. Test locally on the GPU machine:

```bash
curl http://127.0.0.1:8200/health
curl http://127.0.0.1:8300/health
```

Both responses must contain `"ready": true`.

## Connect the GPU machine to Azure

Run this on the GPU machine and keep it running:

```bash
ssh -i idraak_key.pem \
  -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -R 8200:127.0.0.1:8200 \
  -R 8300:127.0.0.1:8300 \
  azureuser@4.232.128.6
```

ASR port 9009 and OCR port 8100 can be included in the same command when those
services are running on this GPU machine.

## Run the pipeline on Azure

Install only the CPU/client dependencies:

```bash
cd video_rag_local
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.azure.txt
sudo apt-get install -y tesseract-ocr ffmpeg
```

Test the tunneled endpoints:

```bash
curl http://127.0.0.1:8200/health
curl http://127.0.0.1:8300/health
```

Then run:

```bash
export XCLIP_URL=http://127.0.0.1:8200
export CAPTION_URL=http://127.0.0.1:8300
export ASR_URL=http://127.0.0.1:9009
export OCR_URL=http://127.0.0.1:8100

python video_pipeline.py --video /path/to/video.mp4 --out_dir output_results
```

If the Azure pipeline itself runs inside Docker, use host networking so its
`127.0.0.1` reaches the SSH tunnel bound on the Azure host:

```bash
docker run --network host ...
```

## API examples

Caption an image:

```bash
curl -X POST http://127.0.0.1:8300/v1/caption \
  -F 'image=@frame.jpg' \
  -F 'prompt=Describe the image in detail.'
```

XCLIP expects one or more multipart fields named `frames` plus `labels`, which
is a JSON array. The pipeline client creates this request automatically.
