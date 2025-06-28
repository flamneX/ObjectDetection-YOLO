{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Cammy276/ObjectDetection-YOLO/blob/main/YOLO11.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "nlYlvlkfGS7F",
        "outputId": "b14ae767-0de5-4deb-eedf-28cdbd3f744f"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Ultralytics 8.3.135 🚀 Python-3.11.12 torch-2.6.0+cu124 CUDA:0 (Tesla T4, 15095MiB)\n",
            "Setup complete ✅ (2 CPUs, 12.7 GB RAM, 41.7/112.6 GB disk)\n",
            "Collecting roboflow\n",
            "  Downloading roboflow-1.1.64-py3-none-any.whl.metadata (9.7 kB)\n",
            "Requirement already satisfied: certifi in /usr/local/lib/python3.11/dist-packages (from roboflow) (2025.4.26)\n",
            "Collecting idna==3.7 (from roboflow)\n",
            "  Downloading idna-3.7-py3-none-any.whl.metadata (9.9 kB)\n",
            "Requirement already satisfied: cycler in /usr/local/lib/python3.11/dist-packages (from roboflow) (0.12.1)\n",
            "Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.11/dist-packages (from roboflow) (1.4.8)\n",
            "Requirement already satisfied: matplotlib in /usr/local/lib/python3.11/dist-packages (from roboflow) (3.10.0)\n",
            "Requirement already satisfied: numpy>=1.18.5 in /usr/local/lib/python3.11/dist-packages (from roboflow) (2.0.2)\n",
            "Collecting opencv-python-headless==4.10.0.84 (from roboflow)\n",
            "  Downloading opencv_python_headless-4.10.0.84-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)\n",
            "Requirement already satisfied: Pillow>=7.1.2 in /usr/local/lib/python3.11/dist-packages (from roboflow) (11.2.1)\n",
            "Collecting pillow-heif>=0.18.0 (from roboflow)\n",
            "  Downloading pillow_heif-0.22.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (9.6 kB)\n",
            "Requirement already satisfied: python-dateutil in /usr/local/lib/python3.11/dist-packages (from roboflow) (2.9.0.post0)\n",
            "Collecting python-dotenv (from roboflow)\n",
            "  Downloading python_dotenv-1.1.0-py3-none-any.whl.metadata (24 kB)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.11/dist-packages (from roboflow) (2.32.3)\n",
            "Requirement already satisfied: six in /usr/local/lib/python3.11/dist-packages (from roboflow) (1.17.0)\n",
            "Requirement already satisfied: urllib3>=1.26.6 in /usr/local/lib/python3.11/dist-packages (from roboflow) (2.4.0)\n",
            "Requirement already satisfied: tqdm>=4.41.0 in /usr/local/lib/python3.11/dist-packages (from roboflow) (4.67.1)\n",
            "Requirement already satisfied: PyYAML>=5.3.1 in /usr/local/lib/python3.11/dist-packages (from roboflow) (6.0.2)\n",
            "Requirement already satisfied: requests-toolbelt in /usr/local/lib/python3.11/dist-packages (from roboflow) (1.0.0)\n",
            "Collecting filetype (from roboflow)\n",
            "  Downloading filetype-1.2.0-py2.py3-none-any.whl.metadata (6.5 kB)\n",
            "Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib->roboflow) (1.3.2)\n",
            "Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib->roboflow) (4.57.0)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib->roboflow) (24.2)\n",
            "Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib->roboflow) (3.2.3)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests->roboflow) (3.4.2)\n",
            "Downloading roboflow-1.1.64-py3-none-any.whl (85 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m85.4/85.4 kB\u001b[0m \u001b[31m6.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading idna-3.7-py3-none-any.whl (66 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m66.8/66.8 kB\u001b[0m \u001b[31m7.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading opencv_python_headless-4.10.0.84-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (49.9 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m49.9/49.9 MB\u001b[0m \u001b[31m20.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading pillow_heif-0.22.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (7.8 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m7.8/7.8 MB\u001b[0m \u001b[31m129.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading filetype-1.2.0-py2.py3-none-any.whl (19 kB)\n",
            "Downloading python_dotenv-1.1.0-py3-none-any.whl (20 kB)\n",
            "Installing collected packages: filetype, python-dotenv, pillow-heif, opencv-python-headless, idna, roboflow\n",
            "  Attempting uninstall: opencv-python-headless\n",
            "    Found existing installation: opencv-python-headless 4.11.0.86\n",
            "    Uninstalling opencv-python-headless-4.11.0.86:\n",
            "      Successfully uninstalled opencv-python-headless-4.11.0.86\n",
            "  Attempting uninstall: idna\n",
            "    Found existing installation: idna 3.10\n",
            "    Uninstalling idna-3.10:\n",
            "      Successfully uninstalled idna-3.10\n",
            "Successfully installed filetype-1.2.0 idna-3.7 opencv-python-headless-4.10.0.84 pillow-heif-0.22.0 python-dotenv-1.1.0 roboflow-1.1.64\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "application/vnd.colab-display-data+json": {
              "pip_warning": {
                "packages": [
                  "cv2"
                ]
              },
              "id": "84f8d510db5f40ac8d8e8491b616e444"
            }
          },
          "metadata": {}
        }
      ],
      "source": [
        "!pip install ultralytics\n",
        "import ultralytics\n",
        "ultralytics.checks()\n",
        "from ultralytics import YOLO\n",
        "from IPython.display import Image\n",
        "\n",
        "!pip install roboflow\n",
        "from roboflow import Roboflow"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "rf = Roboflow(api_key=\"BuDn3mPWk1426lqrnQRc\")\n",
        "project = rf.workspace(\"test-vivys\").project(\"tumor-detection-ko5jp-4h325\")\n",
        "version = project.version(1)\n",
        "dataset = version.download(\"yolov11\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8dyOAbjiGaBS",
        "outputId": "ff9f1c91-f18a-441b-bd1b-1f9e869c5236"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "loading Roboflow workspace...\n",
            "loading Roboflow project...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Downloading Dataset Version Zip in Tumor-Detection-1 to yolov11:: 100%|██████████| 175216/175216 [00:02<00:00, 60814.56it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n",
            "Extracting Dataset Version Zip to Tumor-Detection-1 in yolov11:: 100%|██████████| 9403/9403 [00:01<00:00, 7413.26it/s]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Load the YOLO11n model\n",
        "model = YOLO(\"yolo11n.pt\")\n",
        "\n",
        "# Train the model\n",
        "model.train(\n",
        "    data=f\"{dataset.location}/data.yaml\",  # Dataset path\n",
        "    epochs=50,\n",
        "    batch=16,\n",
        "    imgsz=640,\n",
        "    workers=8,\n",
        "    optimizer=\"auto\",\n",
        "    lr0=0.001,\n",
        "    lrf=0.00001,\n",
        "    momentum=0.937,\n",
        "    box=0.2,\n",
        "    degrees=0.0,\n",
        "    translate=0.1,\n",
        "    shear=0.0,\n",
        "    perspective=0.0,\n",
        "    resume=False,\n",
        "    project=\"yolo11_original\",\n",
        "    name=\"exp\"\n",
        ")\n",
        "\n",
        "# Save path to original weights\n",
        "original_11_path = \"yolo11_original/exp/weights/best.pt\""
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fPlx7d2vG4cA",
        "outputId": "438728b7-21dd-4329-f9ef-c7560a3a3052"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt to 'yolo11n.pt'...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 5.35M/5.35M [00:00<00:00, 91.2MB/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Ultralytics 8.3.135 🚀 Python-3.11.12 torch-2.6.0+cu124 CUDA:0 (Tesla T4, 15095MiB)\n",
            "\u001b[34m\u001b[1mengine/trainer: \u001b[0magnostic_nms=False, amp=True, augment=False, auto_augment=randaugment, batch=16, bgr=0.0, box=0.2, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=/content/Tumor-Detection-1/data.yaml, degrees=0.0, deterministic=True, device=None, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, epochs=50, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.001, lrf=1e-05, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolo11n.pt, momentum=0.937, mosaic=1.0, multi_scale=False, name=exp, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=100, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=yolo11_original, rect=False, resume=False, retina_masks=False, save=True, save_conf=False, save_crop=False, save_dir=yolo11_original/exp, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None\n",
            "Downloading https://ultralytics.com/assets/Arial.ttf to '/root/.config/Ultralytics/Arial.ttf'...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 755k/755k [00:00<00:00, 28.6MB/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Overriding model.yaml nc=80 with nc=5\n",
            "\n",
            "                   from  n    params  module                                       arguments                     \n",
            "  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 \n",
            "  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                \n",
            "  2                  -1  1      6640  ultralytics.nn.modules.block.C3k2            [32, 64, 1, False, 0.25]      \n",
            "  3                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                \n",
            "  4                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]     \n",
            "  5                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              \n",
            "  6                  -1  1     87040  ultralytics.nn.modules.block.C3k2            [128, 128, 1, True]           \n",
            "  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              \n",
            "  8                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]           \n",
            "  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 \n",
            " 10                  -1  1    249728  ultralytics.nn.modules.block.C2PSA           [256, 256, 1]                 \n",
            " 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          \n",
            " 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
            " 13                  -1  1    111296  ultralytics.nn.modules.block.C3k2            [384, 128, 1, False]          \n",
            " 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          \n",
            " 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
            " 16                  -1  1     32096  ultralytics.nn.modules.block.C3k2            [256, 64, 1, False]           \n",
            " 17                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                \n",
            " 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
            " 19                  -1  1     86720  ultralytics.nn.modules.block.C3k2            [192, 128, 1, False]          \n",
            " 20                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              \n",
            " 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
            " 22                  -1  1    378880  ultralytics.nn.modules.block.C3k2            [384, 256, 1, True]           \n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            " 23        [16, 19, 22]  1    431647  ultralytics.nn.modules.head.Detect           [5, [64, 128, 256]]           \n",
            "YOLO11n summary: 181 layers, 2,590,815 parameters, 2,590,799 gradients, 6.4 GFLOPs\n",
            "\n",
            "Transferred 448/499 items from pretrained weights\n",
            "Freezing layer 'model.23.dfl.conv.weight'\n",
            "\u001b[34m\u001b[1mAMP: \u001b[0mrunning Automatic Mixed Precision (AMP) checks...\n",
            "\u001b[34m\u001b[1mAMP: \u001b[0mchecks passed ✅\n",
            "\u001b[34m\u001b[1mtrain: \u001b[0mFast image access ✅ (ping: 0.0±0.0 ms, read: 1100.0±695.7 MB/s, size: 35.1 KB)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\u001b[34m\u001b[1mtrain: \u001b[0mScanning /content/Tumor-Detection-1/train/labels... 4110 images, 0 backgrounds, 0 corrupt: 100%|██████████| 4110/4110 [00:01<00:00, 2148.90it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[34m\u001b[1mtrain: \u001b[0m/content/Tumor-Detection-1/train/images/no_tumor_914_jpg.rf.98383504307f0f063d1774f5ff285583.jpg: 1 duplicate labels removed\n",
            "\u001b[34m\u001b[1mtrain: \u001b[0m/content/Tumor-Detection-1/train/images/no_tumor_914_jpg.rf.af4cf1279633a51344a0f073e7dcb139.jpg: 1 duplicate labels removed\n",
            "\u001b[34m\u001b[1mtrain: \u001b[0m/content/Tumor-Detection-1/train/images/no_tumor_914_jpg.rf.eb30995774830ddbee0d324e034fac18.jpg: 1 duplicate labels removed\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[34m\u001b[1mtrain: \u001b[0mNew cache created: /content/Tumor-Detection-1/train/labels.cache\n",
            "WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = 4367, len(boxes) = 4382. To resolve this only boxes will be used and all segments will be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.\n",
            "\u001b[34m\u001b[1malbumentations: \u001b[0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))\n",
            "\u001b[34m\u001b[1mval: \u001b[0mFast image access ✅ (ping: 0.0±0.0 ms, read: 546.2±365.0 MB/s, size: 31.7 KB)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\u001b[34m\u001b[1mval: \u001b[0mScanning /content/Tumor-Detection-1/valid/labels... 395 images, 0 backgrounds, 0 corrupt: 100%|██████████| 395/395 [00:00<00:00, 1154.64it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[34m\u001b[1mval: \u001b[0mNew cache created: /content/Tumor-Detection-1/valid/labels.cache\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Plotting labels to yolo11_original/exp/labels.jpg... \n",
            "\u001b[34m\u001b[1moptimizer:\u001b[0m 'optimizer=auto' found, ignoring 'lr0=0.001' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... \n",
            "\u001b[34m\u001b[1moptimizer:\u001b[0m AdamW(lr=0.001111, momentum=0.9) with parameter groups 81 weight(decay=0.0), 88 weight(decay=0.0005), 87 bias(decay=0.0)\n",
            "Image sizes 640 train, 640 val\n",
            "Using 2 dataloader workers\n",
            "Logging results to \u001b[1myolo11_original/exp\u001b[0m\n",
            "Starting training for 50 epochs...\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "       1/50      2.28G    0.02425       2.32      1.265         27        640: 100%|██████████| 257/257 [01:17<00:00,  3.34it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:04<00:00,  3.10it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.737      0.412      0.463      0.318\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "       2/50      2.68G    0.02515      1.478      1.273         29        640: 100%|██████████| 257/257 [01:12<00:00,  3.54it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.90it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.801      0.505      0.525      0.384\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "       3/50       2.7G    0.02528      1.173      1.275         28        640: 100%|██████████| 257/257 [01:11<00:00,  3.59it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:04<00:00,  3.05it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.715      0.512      0.554        0.4\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "       4/50      2.71G    0.02493      1.034      1.261         33        640: 100%|██████████| 257/257 [01:10<00:00,  3.63it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.54it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415       0.74      0.505      0.538      0.411\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "       5/50      2.72G    0.02401      0.935      1.244         24        640: 100%|██████████| 257/257 [01:10<00:00,  3.65it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  4.25it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.792      0.576      0.598      0.455\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "       6/50      2.74G    0.02406     0.8963      1.237         22        640: 100%|██████████| 257/257 [01:10<00:00,  3.64it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:02<00:00,  4.40it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.842      0.589       0.64      0.477\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "       7/50      2.75G    0.02284     0.8279      1.211         23        640: 100%|██████████| 257/257 [01:10<00:00,  3.65it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.60it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.819      0.605      0.638      0.487\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "       8/50      2.77G    0.02252     0.7775      1.205         33        640: 100%|██████████| 257/257 [01:10<00:00,  3.65it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.51it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.875      0.566      0.631      0.478\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "       9/50      2.78G    0.02208     0.7627      1.192         29        640: 100%|██████████| 257/257 [01:12<00:00,  3.57it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:02<00:00,  4.51it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.744      0.597      0.606      0.466\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      10/50       2.8G    0.02229     0.7381      1.197         23        640: 100%|██████████| 257/257 [01:11<00:00,  3.59it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:02<00:00,  4.35it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.833      0.573      0.612      0.478\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      11/50      2.81G    0.02149     0.6984      1.174         22        640: 100%|██████████| 257/257 [01:11<00:00,  3.58it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:02<00:00,  4.38it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.869      0.562      0.626      0.489\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      12/50      2.83G    0.02104     0.6765      1.164         23        640: 100%|██████████| 257/257 [01:12<00:00,  3.56it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.86it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.814      0.627      0.637      0.489\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      13/50      2.84G    0.02068      0.677      1.158         32        640: 100%|██████████| 257/257 [01:12<00:00,  3.56it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.47it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.814      0.638      0.651      0.504\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      14/50      2.86G    0.02059      0.659      1.153         21        640: 100%|██████████| 257/257 [01:12<00:00,  3.56it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.72it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.869      0.601      0.641      0.502\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      15/50      2.87G    0.01993     0.6348      1.143         27        640: 100%|██████████| 257/257 [01:11<00:00,  3.60it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:02<00:00,  4.34it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.876      0.604      0.643      0.499\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      16/50      2.89G    0.02029      0.623      1.146         27        640: 100%|██████████| 257/257 [01:12<00:00,  3.57it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:02<00:00,  4.42it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.875      0.597      0.647      0.503\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      17/50       2.9G    0.02002     0.6197      1.133         20        640: 100%|██████████| 257/257 [01:11<00:00,  3.58it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  4.09it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.846      0.629      0.652      0.513\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      18/50      2.92G    0.01968     0.5969      1.131         30        640: 100%|██████████| 257/257 [01:10<00:00,  3.63it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.57it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.857      0.625      0.655      0.523\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      19/50      2.93G    0.01964     0.5959      1.125         24        640: 100%|██████████| 257/257 [01:12<00:00,  3.56it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.47it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.888      0.626      0.654      0.516\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      20/50      2.95G    0.01921     0.5748      1.119         28        640: 100%|██████████| 257/257 [01:10<00:00,  3.62it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:02<00:00,  4.42it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.839      0.627      0.645      0.519\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      21/50      2.96G    0.01883     0.5552      1.104         26        640: 100%|██████████| 257/257 [01:11<00:00,  3.58it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:02<00:00,  4.35it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.895      0.593      0.643      0.518\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      22/50      2.98G    0.01885     0.5571      1.115         25        640: 100%|██████████| 257/257 [01:10<00:00,  3.62it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  4.15it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.678      0.611      0.646      0.518\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      23/50      2.99G    0.01856     0.5358      1.105         34        640: 100%|██████████| 257/257 [01:09<00:00,  3.68it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.40it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.895      0.595      0.642      0.518\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      24/50      3.01G    0.01837     0.5219      1.097         21        640: 100%|██████████| 257/257 [01:10<00:00,  3.67it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:02<00:00,  4.43it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.626      0.619       0.63      0.499\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      25/50      3.02G    0.01778     0.5117      1.084         34        640: 100%|██████████| 257/257 [01:12<00:00,  3.55it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  4.27it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.625      0.697      0.669      0.536\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      26/50      3.04G    0.01859      0.515      1.098         28        640: 100%|██████████| 257/257 [01:15<00:00,  3.41it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  4.31it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.877      0.631      0.665      0.535\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      27/50      3.05G    0.01786     0.5087      1.086         35        640: 100%|██████████| 257/257 [01:12<00:00,  3.57it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  4.25it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.881       0.61      0.654       0.53\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      28/50      3.06G    0.01776     0.4916      1.082         26        640: 100%|██████████| 257/257 [01:10<00:00,  3.66it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.37it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.847      0.658      0.659      0.529\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      29/50      3.08G    0.01728      0.485      1.071         22        640: 100%|██████████| 257/257 [01:09<00:00,  3.68it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  4.04it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.879      0.641      0.661       0.53\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      30/50       3.1G    0.01727     0.4719      1.078         24        640: 100%|██████████| 257/257 [01:11<00:00,  3.58it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  4.29it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.697      0.621      0.652      0.523\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      31/50      3.11G    0.01668      0.465      1.062         28        640: 100%|██████████| 257/257 [01:13<00:00,  3.49it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  4.25it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.889      0.626      0.665      0.532\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      32/50      3.12G    0.01675     0.4602      1.059         30        640: 100%|██████████| 257/257 [01:14<00:00,  3.46it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  4.14it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.878      0.631      0.648      0.516\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      33/50      3.14G    0.01642     0.4447      1.056         33        640: 100%|██████████| 257/257 [01:12<00:00,  3.55it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  4.06it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.892      0.626      0.672      0.538\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      34/50      3.15G    0.01617     0.4451      1.053         29        640: 100%|██████████| 257/257 [01:12<00:00,  3.54it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.73it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.867      0.639      0.668      0.537\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      35/50      3.17G    0.01596     0.4344      1.043         24        640: 100%|██████████| 257/257 [01:13<00:00,  3.52it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.32it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.914      0.633      0.679      0.539\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      36/50      3.18G    0.01585     0.4268      1.048         22        640: 100%|██████████| 257/257 [01:11<00:00,  3.61it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.47it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.894       0.63      0.673      0.537\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      37/50       3.2G    0.01579     0.4171      1.047         24        640: 100%|██████████| 257/257 [01:12<00:00,  3.55it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  4.03it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.912      0.611      0.651      0.523\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      38/50      3.21G    0.01577     0.4255      1.038         29        640: 100%|██████████| 257/257 [01:12<00:00,  3.57it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:02<00:00,  4.38it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.914      0.623      0.658      0.537\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      39/50      3.23G    0.01548     0.4182      1.032         28        640: 100%|██████████| 257/257 [01:12<00:00,  3.55it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  4.23it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.923      0.619      0.674      0.549\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      40/50      3.24G    0.01508     0.4057      1.028         33        640: 100%|██████████| 257/257 [01:13<00:00,  3.51it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  4.27it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.921      0.612      0.662      0.543\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Closing dataloader mosaic\n",
            "\u001b[34m\u001b[1malbumentations: \u001b[0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      41/50      3.26G     0.0128     0.3291     0.9793         15        640: 100%|██████████| 257/257 [01:10<00:00,  3.62it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.38it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.886       0.64      0.676       0.54\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      42/50      3.28G    0.01233     0.3068     0.9649         15        640: 100%|██████████| 257/257 [01:07<00:00,  3.79it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:02<00:00,  4.49it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.863      0.632      0.646       0.52\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      43/50      3.29G    0.01197     0.2925     0.9514         16        640: 100%|██████████| 257/257 [01:09<00:00,  3.71it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  4.08it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.917      0.607      0.654      0.533\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      44/50       3.3G    0.01166     0.2913     0.9449         14        640: 100%|██████████| 257/257 [01:08<00:00,  3.76it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  4.11it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.915       0.62       0.66      0.534\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      45/50      3.32G    0.01151     0.2851     0.9448         15        640: 100%|██████████| 257/257 [01:08<00:00,  3.77it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  4.12it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.912      0.619      0.655      0.538\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      46/50      3.33G    0.01152     0.2805     0.9399         14        640: 100%|██████████| 257/257 [01:08<00:00,  3.74it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:04<00:00,  3.12it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.872      0.628      0.658      0.528\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      47/50      3.35G    0.01105     0.2693     0.9269         14        640: 100%|██████████| 257/257 [01:07<00:00,  3.78it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:02<00:00,  4.48it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.859      0.631       0.65       0.53\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      48/50      3.36G    0.01093     0.2657     0.9239         15        640: 100%|██████████| 257/257 [01:07<00:00,  3.80it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:04<00:00,  3.25it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.889      0.615       0.65      0.532\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      49/50      3.38G    0.01065     0.2602     0.9187         14        640: 100%|██████████| 257/257 [01:07<00:00,  3.81it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:02<00:00,  4.51it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.864      0.627      0.646       0.53\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      50/50      3.39G    0.01064     0.2595     0.9172         16        640: 100%|██████████| 257/257 [01:08<00:00,  3.77it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  4.16it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.872       0.63      0.649      0.535\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "50 epochs completed in 1.042 hours.\n",
            "Optimizer stripped from yolo11_original/exp/weights/last.pt, 5.5MB\n",
            "Optimizer stripped from yolo11_original/exp/weights/best.pt, 5.5MB\n",
            "\n",
            "Validating yolo11_original/exp/weights/best.pt...\n",
            "Ultralytics 8.3.135 🚀 Python-3.11.12 torch-2.6.0+cu124 CUDA:0 (Tesla T4, 15095MiB)\n",
            "YOLO11n summary (fused): 100 layers, 2,583,127 parameters, 0 gradients, 6.3 GFLOPs\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:04<00:00,  2.66it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.925      0.621      0.676      0.549\n",
            "              NO_tumor        115        116      0.974      0.976      0.983      0.811\n",
            "                glioma         30         36      0.949       0.52      0.655      0.471\n",
            "            meningioma        144        148      0.921      0.905       0.94      0.818\n",
            "             pituitary        106        111      0.781      0.703      0.773      0.625\n",
            "space-occupying lesion-          1          4          1          0     0.0308     0.0219\n",
            "Speed: 0.3ms preprocess, 2.3ms inference, 0.0ms loss, 2.3ms postprocess per image\n",
            "Results saved to \u001b[1myolo11_original/exp\u001b[0m\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Train the fine-tuned model\n",
        "model.train(\n",
        "    data=f\"{dataset.location}/data.yaml\",\n",
        "    epochs=50,\n",
        "    batch=16,\n",
        "    imgsz=640,\n",
        "    workers=8,\n",
        "    optimizer=\"auto\",\n",
        "    lr0=0.01,\n",
        "    lrf=0.01,\n",
        "    momentum=0.937,\n",
        "    box=0.2,\n",
        "    degrees=0.0,\n",
        "    translate=0.1,\n",
        "    shear=0.0,\n",
        "    perspective=0.0,\n",
        "    resume=False,\n",
        "    project=\"yolo11_finetune\",\n",
        "    name=\"exp\"\n",
        ")\n",
        "\n",
        "# Save path to fine-tuned weights\n",
        "finetuned_11_path = \"yolo11_finetune/exp/weights/best.pt\""
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "BiU73hwMHTGM",
        "outputId": "c3822125-07c1-4dc4-83c2-778301fadfc6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Ultralytics 8.3.135 🚀 Python-3.11.12 torch-2.6.0+cu124 CUDA:0 (Tesla T4, 15095MiB)\n",
            "\u001b[34m\u001b[1mengine/trainer: \u001b[0magnostic_nms=False, amp=True, augment=False, auto_augment=randaugment, batch=16, bgr=0.0, box=0.2, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=/content/Tumor-Detection-1/data.yaml, degrees=0.0, deterministic=True, device=None, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, epochs=50, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolo11n.pt, momentum=0.937, mosaic=1.0, multi_scale=False, name=exp, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=100, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=yolo11_finetune, rect=False, resume=False, retina_masks=False, save=True, save_conf=False, save_crop=False, save_dir=yolo11_finetune/exp, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.0, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None\n",
            "\n",
            "                   from  n    params  module                                       arguments                     \n",
            "  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 \n",
            "  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                \n",
            "  2                  -1  1      6640  ultralytics.nn.modules.block.C3k2            [32, 64, 1, False, 0.25]      \n",
            "  3                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                \n",
            "  4                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]     \n",
            "  5                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              \n",
            "  6                  -1  1     87040  ultralytics.nn.modules.block.C3k2            [128, 128, 1, True]           \n",
            "  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              \n",
            "  8                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]           \n",
            "  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 \n",
            " 10                  -1  1    249728  ultralytics.nn.modules.block.C2PSA           [256, 256, 1]                 \n",
            " 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          \n",
            " 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
            " 13                  -1  1    111296  ultralytics.nn.modules.block.C3k2            [384, 128, 1, False]          \n",
            " 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          \n",
            " 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
            " 16                  -1  1     32096  ultralytics.nn.modules.block.C3k2            [256, 64, 1, False]           \n",
            " 17                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                \n",
            " 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
            " 19                  -1  1     86720  ultralytics.nn.modules.block.C3k2            [192, 128, 1, False]          \n",
            " 20                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              \n",
            " 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
            " 22                  -1  1    378880  ultralytics.nn.modules.block.C3k2            [384, 256, 1, True]           \n",
            " 23        [16, 19, 22]  1    431647  ultralytics.nn.modules.head.Detect           [5, [64, 128, 256]]           \n",
            "YOLO11n summary: 181 layers, 2,590,815 parameters, 2,590,799 gradients, 6.4 GFLOPs\n",
            "\n",
            "Transferred 499/499 items from pretrained weights\n",
            "Freezing layer 'model.23.dfl.conv.weight'\n",
            "\u001b[34m\u001b[1mAMP: \u001b[0mrunning Automatic Mixed Precision (AMP) checks...\n",
            "\u001b[34m\u001b[1mAMP: \u001b[0mchecks passed ✅\n",
            "\u001b[34m\u001b[1mtrain: \u001b[0mFast image access ✅ (ping: 0.0±0.0 ms, read: 1011.6±333.9 MB/s, size: 35.1 KB)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\u001b[34m\u001b[1mtrain: \u001b[0mScanning /content/Tumor-Detection-1/train/labels.cache... 4110 images, 0 backgrounds, 0 corrupt: 100%|██████████| 4110/4110 [00:00<?, ?it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[34m\u001b[1mtrain: \u001b[0m/content/Tumor-Detection-1/train/images/no_tumor_914_jpg.rf.98383504307f0f063d1774f5ff285583.jpg: 1 duplicate labels removed\n",
            "\u001b[34m\u001b[1mtrain: \u001b[0m/content/Tumor-Detection-1/train/images/no_tumor_914_jpg.rf.af4cf1279633a51344a0f073e7dcb139.jpg: 1 duplicate labels removed\n",
            "\u001b[34m\u001b[1mtrain: \u001b[0m/content/Tumor-Detection-1/train/images/no_tumor_914_jpg.rf.eb30995774830ddbee0d324e034fac18.jpg: 1 duplicate labels removed\n",
            "WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = 4367, len(boxes) = 4382. To resolve this only boxes will be used and all segments will be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.\n",
            "\u001b[34m\u001b[1malbumentations: \u001b[0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[34m\u001b[1mval: \u001b[0mFast image access ✅ (ping: 0.0±0.0 ms, read: 454.0±322.7 MB/s, size: 31.7 KB)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\u001b[34m\u001b[1mval: \u001b[0mScanning /content/Tumor-Detection-1/valid/labels.cache... 395 images, 0 backgrounds, 0 corrupt: 100%|██████████| 395/395 [00:00<?, ?it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Plotting labels to yolo11_finetune/exp/labels.jpg... \n",
            "\u001b[34m\u001b[1moptimizer:\u001b[0m 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... \n",
            "\u001b[34m\u001b[1moptimizer:\u001b[0m AdamW(lr=0.001111, momentum=0.9) with parameter groups 81 weight(decay=0.0), 88 weight(decay=0.0005), 87 bias(decay=0.0)\n",
            "Image sizes 640 train, 640 val\n",
            "Using 2 dataloader workers\n",
            "Logging results to \u001b[1myolo11_finetune/exp\u001b[0m\n",
            "Starting training for 50 epochs...\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "       1/50      2.63G    0.01546      0.411      1.035         27        640: 100%|██████████| 257/257 [01:17<00:00,  3.30it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:02<00:00,  4.37it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.881       0.61       0.65      0.522\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "       2/50      2.79G    0.01701     0.4724      1.064         29        640: 100%|██████████| 257/257 [01:13<00:00,  3.50it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.82it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.627      0.589      0.613      0.481\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "       3/50      2.79G    0.01796     0.5118      1.081         28        640: 100%|██████████| 257/257 [01:12<00:00,  3.54it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.45it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415       0.63      0.623      0.638      0.489\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "       4/50      2.79G    0.01861     0.5575      1.099         33        640: 100%|██████████| 257/257 [01:12<00:00,  3.56it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.40it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.844      0.622      0.639      0.493\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "       5/50      2.79G    0.01842     0.5348      1.093         24        640: 100%|██████████| 257/257 [01:13<00:00,  3.49it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.42it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.634      0.606      0.639      0.498\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "       6/50      2.79G    0.01872     0.5485      1.101         22        640: 100%|██████████| 257/257 [01:11<00:00,  3.57it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.43it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.631      0.613      0.627      0.487\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "       7/50      2.79G    0.01819     0.5211      1.089         23        640: 100%|██████████| 257/257 [01:12<00:00,  3.56it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.59it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.829      0.575      0.617      0.484\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "       8/50      2.79G    0.01834     0.5194      1.098         33        640: 100%|██████████| 257/257 [01:11<00:00,  3.58it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.36it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.841      0.617       0.65      0.508\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "       9/50      2.79G    0.01826     0.5094      1.089         29        640: 100%|██████████| 257/257 [01:12<00:00,  3.57it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.32it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.847      0.638      0.637      0.506\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      10/50      2.79G    0.01835     0.5096      1.096         23        640: 100%|██████████| 257/257 [01:13<00:00,  3.49it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.68it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.631      0.628      0.633      0.505\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      11/50      2.79G    0.01745     0.4791      1.074         22        640: 100%|██████████| 257/257 [01:11<00:00,  3.60it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.73it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.841      0.609      0.631      0.507\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      12/50      2.79G    0.01754     0.4808      1.077         23        640: 100%|██████████| 257/257 [01:11<00:00,  3.59it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.25it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.873      0.619      0.649      0.514\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      13/50      2.79G    0.01724     0.4892      1.072         32        640: 100%|██████████| 257/257 [01:12<00:00,  3.56it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.46it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415       0.83      0.618      0.631      0.498\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      14/50      2.79G    0.01712     0.4798      1.066         21        640: 100%|██████████| 257/257 [01:11<00:00,  3.60it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.39it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.627      0.648      0.638      0.514\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      15/50      2.79G    0.01691     0.4647      1.063         27        640: 100%|██████████| 257/257 [01:10<00:00,  3.63it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.59it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.689      0.618       0.65      0.519\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      16/50      2.79G    0.01693      0.466      1.066         27        640: 100%|██████████| 257/257 [01:12<00:00,  3.53it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.29it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.644      0.632      0.639      0.514\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      17/50      2.79G    0.01712     0.4632      1.061         20        640: 100%|██████████| 257/257 [01:11<00:00,  3.59it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.42it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.571      0.608      0.595      0.478\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      18/50      2.79G     0.0166     0.4544      1.056         30        640: 100%|██████████| 257/257 [01:11<00:00,  3.62it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.55it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.866      0.621      0.644       0.52\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      19/50      2.79G    0.01653     0.4503      1.052         24        640: 100%|██████████| 257/257 [01:11<00:00,  3.62it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.77it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.847      0.619      0.632      0.508\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      20/50      2.79G    0.01628     0.4397      1.048         28        640: 100%|██████████| 257/257 [01:10<00:00,  3.63it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:02<00:00,  4.35it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.912      0.621       0.65      0.511\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      21/50      2.79G    0.01605     0.4253      1.038         26        640: 100%|██████████| 257/257 [01:12<00:00,  3.53it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.66it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415       0.65      0.619      0.636      0.518\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      22/50      2.79G     0.0161     0.4339      1.044         25        640: 100%|██████████| 257/257 [01:12<00:00,  3.55it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  4.27it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.885      0.606      0.649      0.518\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      23/50      2.79G     0.0158     0.4165      1.039         34        640: 100%|██████████| 257/257 [01:13<00:00,  3.50it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  4.31it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.845      0.625      0.631      0.515\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      24/50      2.79G    0.01563     0.4123       1.03         21        640: 100%|██████████| 257/257 [01:12<00:00,  3.54it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.60it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415       0.65      0.636      0.638       0.51\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      25/50      2.79G     0.0153      0.406      1.027         34        640: 100%|██████████| 257/257 [01:12<00:00,  3.53it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:02<00:00,  4.34it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415       0.63      0.639       0.64      0.516\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      26/50      2.79G    0.01586     0.4132      1.034         28        640: 100%|██████████| 257/257 [01:14<00:00,  3.45it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.41it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.627       0.67      0.653      0.535\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      27/50      2.79G    0.01514     0.4049      1.024         35        640: 100%|██████████| 257/257 [01:14<00:00,  3.46it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.69it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.661      0.626       0.64       0.52\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      28/50      2.79G    0.01515     0.3961      1.021         26        640: 100%|██████████| 257/257 [01:10<00:00,  3.62it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.35it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415       0.87      0.604       0.63      0.519\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      29/50      2.79G    0.01491     0.3878      1.019         22        640: 100%|██████████| 257/257 [01:10<00:00,  3.63it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.44it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415       0.65      0.653       0.64      0.527\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      30/50      2.79G    0.01474     0.3824      1.018         24        640: 100%|██████████| 257/257 [01:11<00:00,  3.58it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:04<00:00,  3.20it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.679      0.643      0.654      0.531\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      31/50      2.79G    0.01456     0.3819       1.01         28        640: 100%|██████████| 257/257 [01:11<00:00,  3.61it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.62it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.665      0.625      0.639      0.519\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      32/50      2.79G    0.01443     0.3745       1.01         30        640: 100%|██████████| 257/257 [01:12<00:00,  3.57it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.63it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.649      0.609      0.621      0.512\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      33/50      2.79G    0.01426     0.3641      1.008         33        640: 100%|██████████| 257/257 [01:11<00:00,  3.58it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:04<00:00,  3.11it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.866      0.637      0.646       0.53\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      34/50      2.79G    0.01417      0.369      1.006         29        640: 100%|██████████| 257/257 [01:12<00:00,  3.54it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.44it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.665      0.632      0.643       0.53\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      35/50      2.79G    0.01388     0.3643     0.9971         24        640: 100%|██████████| 257/257 [01:10<00:00,  3.64it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.57it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.875      0.629      0.645      0.533\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      36/50      2.79G    0.01384     0.3623          1         22        640: 100%|██████████| 257/257 [01:10<00:00,  3.63it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  4.20it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.903      0.599      0.645       0.53\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      37/50      2.79G    0.01372     0.3486          1         24        640: 100%|██████████| 257/257 [01:13<00:00,  3.50it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.82it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.869      0.648      0.651      0.528\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      38/50      2.79G    0.01387     0.3559     0.9934         29        640: 100%|██████████| 257/257 [01:13<00:00,  3.51it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:04<00:00,  3.24it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.886      0.613      0.642      0.534\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      39/50      2.79G    0.01362     0.3538     0.9895         28        640: 100%|██████████| 257/257 [01:11<00:00,  3.60it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.41it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.916      0.607      0.656      0.541\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      40/50      2.79G    0.01355     0.3504     0.9913         33        640: 100%|██████████| 257/257 [01:12<00:00,  3.54it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.39it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.894      0.625      0.652      0.536\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Closing dataloader mosaic\n",
            "\u001b[34m\u001b[1malbumentations: \u001b[0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      41/50      2.79G    0.01129     0.2756     0.9358         15        640: 100%|██████████| 257/257 [01:09<00:00,  3.68it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  4.04it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.653      0.662      0.658      0.533\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      42/50      2.79G    0.01107     0.2631     0.9271         15        640: 100%|██████████| 257/257 [01:07<00:00,  3.79it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  4.27it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.874      0.638      0.654      0.536\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      43/50      2.79G    0.01074     0.2535      0.917         16        640: 100%|██████████| 257/257 [01:10<00:00,  3.64it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.78it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.894      0.619      0.654       0.54\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      44/50      2.79G    0.01046     0.2509     0.9086         14        640: 100%|██████████| 257/257 [01:07<00:00,  3.79it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  4.15it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.867      0.642      0.654      0.542\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      45/50      2.79G    0.01033     0.2464     0.9123         15        640: 100%|██████████| 257/257 [01:08<00:00,  3.76it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:02<00:00,  4.43it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.865      0.645      0.647      0.537\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      46/50      2.79G    0.01024     0.2432     0.9063         14        640: 100%|██████████| 257/257 [01:08<00:00,  3.77it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.82it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.877      0.631      0.652      0.537\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      47/50      2.79G    0.01002     0.2374     0.8984         14        640: 100%|██████████| 257/257 [01:07<00:00,  3.82it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:02<00:00,  4.53it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415       0.88       0.63      0.658      0.548\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      48/50      2.79G   0.009838     0.2346     0.8961         15        640: 100%|██████████| 257/257 [01:09<00:00,  3.71it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:02<00:00,  4.37it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.866      0.649      0.667      0.554\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      49/50      2.79G   0.009737     0.2308     0.8945         14        640: 100%|██████████| 257/257 [01:07<00:00,  3.81it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  3.62it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.853      0.638      0.656      0.546\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      50/50      2.79G   0.009666     0.2293     0.8909         16        640: 100%|██████████| 257/257 [01:06<00:00,  3.85it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:03<00:00,  4.20it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.852      0.649      0.659      0.548\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "50 epochs completed in 1.050 hours.\n",
            "Optimizer stripped from yolo11_finetune/exp/weights/last.pt, 5.5MB\n",
            "Optimizer stripped from yolo11_finetune/exp/weights/best.pt, 5.5MB\n",
            "\n",
            "Validating yolo11_finetune/exp/weights/best.pt...\n",
            "Ultralytics 8.3.135 🚀 Python-3.11.12 torch-2.6.0+cu124 CUDA:0 (Tesla T4, 15095MiB)\n",
            "YOLO11n summary (fused): 100 layers, 2,583,127 parameters, 0 gradients, 6.3 GFLOPs\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:05<00:00,  2.28it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.866      0.649      0.666      0.553\n",
            "              NO_tumor        115        116      0.919      0.979       0.97      0.809\n",
            "                glioma         30         36      0.674      0.573       0.65      0.499\n",
            "            meningioma        144        148       0.94      0.919       0.95      0.834\n",
            "             pituitary        106        111      0.799      0.775      0.762      0.625\n",
            "space-occupying lesion-          1          4          1          0          0          0\n",
            "Speed: 0.3ms preprocess, 2.7ms inference, 0.0ms loss, 3.8ms postprocess per image\n",
            "Results saved to \u001b[1myolo11_finetune/exp\u001b[0m\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# === Load original and fine-tuned models ===\n",
        "original_11_model = YOLO(original_11_path)\n",
        "fine_tuned_11_model = YOLO(finetuned_11_path)"
      ],
      "metadata": {
        "id": "fcDXUoJeHaCm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# === TRAINING SET ===\n",
        "results_original_11_train = original_11_model.val(data=f\"{dataset.location}/data.yaml\", split='train', save_conf=True, plots=True, name='original_11_train')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8tkf6DwkHgxg",
        "outputId": "ee43857e-b2e0-4e1a-e8c6-8fc2a88ad018"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Ultralytics 8.3.135 🚀 Python-3.11.12 torch-2.6.0+cu124 CUDA:0 (Tesla T4, 15095MiB)\n",
            "YOLO11n summary (fused): 100 layers, 2,583,127 parameters, 0 gradients, 6.3 GFLOPs\n",
            "\u001b[34m\u001b[1mval: \u001b[0mFast image access ✅ (ping: 0.0±0.0 ms, read: 1141.7±590.8 MB/s, size: 33.3 KB)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\u001b[34m\u001b[1mval: \u001b[0mScanning /content/Tumor-Detection-1/train/labels.cache... 4110 images, 0 backgrounds, 0 corrupt: 100%|██████████| 4110/4110 [00:00<?, ?it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[34m\u001b[1mtrain: \u001b[0m/content/Tumor-Detection-1/train/images/no_tumor_914_jpg.rf.98383504307f0f063d1774f5ff285583.jpg: 1 duplicate labels removed\n",
            "\u001b[34m\u001b[1mtrain: \u001b[0m/content/Tumor-Detection-1/train/images/no_tumor_914_jpg.rf.af4cf1279633a51344a0f073e7dcb139.jpg: 1 duplicate labels removed\n",
            "\u001b[34m\u001b[1mtrain: \u001b[0m/content/Tumor-Detection-1/train/images/no_tumor_914_jpg.rf.eb30995774830ddbee0d324e034fac18.jpg: 1 duplicate labels removed\n",
            "WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = 4367, len(boxes) = 4382. To resolve this only boxes will be used and all segments will be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 257/257 [00:38<00:00,  6.64it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all       4110       4382      0.942      0.907      0.944      0.841\n",
            "              NO_tumor       1101       1110          1      0.995      0.995      0.907\n",
            "                glioma        351        447      0.972      0.904      0.934      0.777\n",
            "            meningioma       1716       1787      0.987      0.976      0.987      0.928\n",
            "             pituitary        942        990       0.99      0.972      0.991        0.9\n",
            "space-occupying lesion-         36         48      0.761      0.688      0.813      0.693\n",
            "Speed: 0.3ms preprocess, 3.7ms inference, 0.0ms loss, 1.4ms postprocess per image\n",
            "Results saved to \u001b[1mruns/detect/original_11_train\u001b[0m\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "results_finetuned_11_train = fine_tuned_11_model.val(data=f\"{dataset.location}/data.yaml\", split='train', save_conf=True, plots=True, name='finetuned_11_train')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Of9TTE4bHzqv",
        "outputId": "6cef5bc6-d3f1-413d-f14a-5423d91ca5d1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Ultralytics 8.3.135 🚀 Python-3.11.12 torch-2.6.0+cu124 CUDA:0 (Tesla T4, 15095MiB)\n",
            "YOLO11n summary (fused): 100 layers, 2,583,127 parameters, 0 gradients, 6.3 GFLOPs\n",
            "\u001b[34m\u001b[1mval: \u001b[0mFast image access ✅ (ping: 0.0±0.0 ms, read: 884.8±461.4 MB/s, size: 33.3 KB)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\u001b[34m\u001b[1mval: \u001b[0mScanning /content/Tumor-Detection-1/train/labels.cache... 4110 images, 0 backgrounds, 0 corrupt: 100%|██████████| 4110/4110 [00:00<?, ?it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[34m\u001b[1mtrain: \u001b[0m/content/Tumor-Detection-1/train/images/no_tumor_914_jpg.rf.98383504307f0f063d1774f5ff285583.jpg: 1 duplicate labels removed\n",
            "\u001b[34m\u001b[1mtrain: \u001b[0m/content/Tumor-Detection-1/train/images/no_tumor_914_jpg.rf.af4cf1279633a51344a0f073e7dcb139.jpg: 1 duplicate labels removed\n",
            "\u001b[34m\u001b[1mtrain: \u001b[0m/content/Tumor-Detection-1/train/images/no_tumor_914_jpg.rf.eb30995774830ddbee0d324e034fac18.jpg: 1 duplicate labels removed\n",
            "WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = 4367, len(boxes) = 4382. To resolve this only boxes will be used and all segments will be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 257/257 [00:37<00:00,  6.93it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all       4110       4382      0.961      0.972      0.985      0.916\n",
            "              NO_tumor       1101       1110      0.999      0.999      0.995      0.969\n",
            "                glioma        351        447      0.998      0.917      0.981      0.867\n",
            "            meningioma       1716       1787       0.99      0.976      0.993      0.955\n",
            "             pituitary        942        990      0.996      0.988      0.995      0.936\n",
            "space-occupying lesion-         36         48      0.823      0.979      0.961      0.855\n",
            "Speed: 0.3ms preprocess, 3.6ms inference, 0.0ms loss, 1.3ms postprocess per image\n",
            "Results saved to \u001b[1mruns/detect/finetuned_11_train\u001b[0m\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# === VALIDATION SET ===\n",
        "results_original_11_val = original_11_model.val(data=f\"{dataset.location}/data.yaml\", split='val', save_conf=True, plots=True, name='original_11_val')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LXhAUS2aH4zj",
        "outputId": "0eb2af32-db3a-47ec-9659-6dd146b1e081"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Ultralytics 8.3.135 🚀 Python-3.11.12 torch-2.6.0+cu124 CUDA:0 (Tesla T4, 15095MiB)\n",
            "\u001b[34m\u001b[1mval: \u001b[0mFast image access ✅ (ping: 0.0±0.0 ms, read: 1169.5±610.2 MB/s, size: 51.2 KB)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\u001b[34m\u001b[1mval: \u001b[0mScanning /content/Tumor-Detection-1/valid/labels.cache... 395 images, 0 backgrounds, 0 corrupt: 100%|██████████| 395/395 [00:00<?, ?it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 25/25 [00:06<00:00,  4.07it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.925      0.621      0.676       0.55\n",
            "              NO_tumor        115        116      0.974      0.976      0.984       0.81\n",
            "                glioma         30         36      0.949      0.521      0.655      0.472\n",
            "            meningioma        144        148      0.921      0.905       0.94      0.818\n",
            "             pituitary        106        111       0.78      0.703      0.772      0.627\n",
            "space-occupying lesion-          1          4          1          0     0.0309     0.0219\n",
            "Speed: 1.8ms preprocess, 4.7ms inference, 0.0ms loss, 2.3ms postprocess per image\n",
            "Results saved to \u001b[1mruns/detect/original_11_val\u001b[0m\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "results_finetuned_11_val = fine_tuned_11_model.val(data=f\"{dataset.location}/data.yaml\", split='val', save_conf=True, plots=True, name='finetuned_11_val')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "iv5E1Hv0IH9W",
        "outputId": "0898a68a-01d5-446a-90ff-0f0fb97245de"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Ultralytics 8.3.135 🚀 Python-3.11.12 torch-2.6.0+cu124 CUDA:0 (Tesla T4, 15095MiB)\n",
            "\u001b[34m\u001b[1mval: \u001b[0mFast image access ✅ (ping: 0.0±0.0 ms, read: 1274.4±675.4 MB/s, size: 51.2 KB)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\u001b[34m\u001b[1mval: \u001b[0mScanning /content/Tumor-Detection-1/valid/labels.cache... 395 images, 0 backgrounds, 0 corrupt: 100%|██████████| 395/395 [00:00<?, ?it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 25/25 [00:06<00:00,  3.91it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        395        415      0.866      0.649      0.667      0.554\n",
            "              NO_tumor        115        116      0.919      0.979       0.97      0.809\n",
            "                glioma         30         36      0.674      0.573      0.651      0.501\n",
            "            meningioma        144        148       0.94      0.919       0.95      0.834\n",
            "             pituitary        106        111      0.798      0.775      0.762      0.625\n",
            "space-occupying lesion-          1          4          1          0          0          0\n",
            "Speed: 1.3ms preprocess, 4.4ms inference, 0.0ms loss, 2.4ms postprocess per image\n",
            "Results saved to \u001b[1mruns/detect/finetuned_11_val\u001b[0m\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# === TESTING SET ===\n",
        "results_original_11_test = original_11_model.val(data=f\"{dataset.location}/data.yaml\", split='test', save_conf=True, plots=True, name='original_11_test')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kU-EZafOIQ5l",
        "outputId": "17c820a0-3e01-4f9c-ed3c-cf15e1fe7c08"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Ultralytics 8.3.135 🚀 Python-3.11.12 torch-2.6.0+cu124 CUDA:0 (Tesla T4, 15095MiB)\n",
            "\u001b[34m\u001b[1mval: \u001b[0mFast image access ✅ (ping: 0.0±0.0 ms, read: 683.6±87.1 MB/s, size: 35.6 KB)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\u001b[34m\u001b[1mval: \u001b[0mScanning /content/Tumor-Detection-1/test/labels... 191 images, 0 backgrounds, 0 corrupt: 100%|██████████| 191/191 [00:00<00:00, 2049.56it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[34m\u001b[1mval: \u001b[0mNew cache created: /content/Tumor-Detection-1/test/labels.cache\n",
            "WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = 201, len(boxes) = 202. To resolve this only boxes will be used and all segments will be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 12/12 [00:04<00:00,  2.97it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        191        202      0.816      0.629      0.637      0.501\n",
            "              NO_tumor         68         68      0.976      0.985      0.993      0.771\n",
            "                glioma         11         14      0.517      0.571      0.513      0.331\n",
            "            meningioma         72         77      0.912      0.922      0.922      0.835\n",
            "             pituitary         40         41      0.678      0.668      0.756      0.567\n",
            "space-occupying lesion-          1          2          1          0          0          0\n",
            "Speed: 2.0ms preprocess, 6.4ms inference, 0.0ms loss, 2.9ms postprocess per image\n",
            "Results saved to \u001b[1mruns/detect/original_11_test\u001b[0m\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "results_finetuned_11_test = fine_tuned_11_model.val(data=f\"{dataset.location}/data.yaml\", split='test', save_conf=True, plots=True, name='finetuned_11_test')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nIqDNQLqIb_E",
        "outputId": "6671700f-a446-474a-f107-0b9a6ddb3b23"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Ultralytics 8.3.135 🚀 Python-3.11.12 torch-2.6.0+cu124 CUDA:0 (Tesla T4, 15095MiB)\n",
            "\u001b[34m\u001b[1mval: \u001b[0mFast image access ✅ (ping: 0.0±0.0 ms, read: 755.8±210.3 MB/s, size: 35.6 KB)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\u001b[34m\u001b[1mval: \u001b[0mScanning /content/Tumor-Detection-1/test/labels.cache... 191 images, 0 backgrounds, 0 corrupt: 100%|██████████| 191/191 [00:00<?, ?it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = 201, len(boxes) = 202. To resolve this only boxes will be used and all segments will be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 12/12 [00:03<00:00,  3.81it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        191        202       0.65      0.643      0.653      0.502\n",
            "              NO_tumor         68         68      0.957      0.991      0.995      0.792\n",
            "                glioma         11         14      0.692      0.643      0.592      0.307\n",
            "            meningioma         72         77      0.948      0.896      0.916      0.836\n",
            "             pituitary         40         41      0.653      0.683      0.762      0.576\n",
            "space-occupying lesion-          1          2          0          0          0          0\n",
            "Speed: 1.8ms preprocess, 5.0ms inference, 0.0ms loss, 2.0ms postprocess per image\n",
            "Results saved to \u001b[1mruns/detect/finetuned_11_test\u001b[0m\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import files\n",
        "\n",
        "import shutil\n",
        "\n",
        "# Zip the folder\n",
        "shutil.make_archive('/content/YOLO11', 'zip', '/content')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "id": "-Hj6DZHPInsD",
        "outputId": "bd6e931b-ecaf-4a29-c1da-2c1a26023ba2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'/content/YOLO11.zip'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "files.download('/content/YOLO11.zip')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "id": "QtEBtp-NIzmV",
        "outputId": "b66b2b58-d7d2-43b0-aaf8-d390f25b6ec9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "    async function download(id, filename, size) {\n",
              "      if (!google.colab.kernel.accessAllowed) {\n",
              "        return;\n",
              "      }\n",
              "      const div = document.createElement('div');\n",
              "      const label = document.createElement('label');\n",
              "      label.textContent = `Downloading \"${filename}\": `;\n",
              "      div.appendChild(label);\n",
              "      const progress = document.createElement('progress');\n",
              "      progress.max = size;\n",
              "      div.appendChild(progress);\n",
              "      document.body.appendChild(div);\n",
              "\n",
              "      const buffers = [];\n",
              "      let downloaded = 0;\n",
              "\n",
              "      const channel = await google.colab.kernel.comms.open(id);\n",
              "      // Send a message to notify the kernel that we're ready.\n",
              "      channel.send({})\n",
              "\n",
              "      for await (const message of channel.messages) {\n",
              "        // Send a message to notify the kernel that we're ready.\n",
              "        channel.send({})\n",
              "        if (message.buffers) {\n",
              "          for (const buffer of message.buffers) {\n",
              "            buffers.push(buffer);\n",
              "            downloaded += buffer.byteLength;\n",
              "            progress.value = downloaded;\n",
              "          }\n",
              "        }\n",
              "      }\n",
              "      const blob = new Blob(buffers, {type: 'application/binary'});\n",
              "      const a = document.createElement('a');\n",
              "      a.href = window.URL.createObjectURL(blob);\n",
              "      a.download = filename;\n",
              "      div.appendChild(a);\n",
              "      a.click();\n",
              "      div.remove();\n",
              "    }\n",
              "  "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "download(\"download_9df11a31-d7d2-4968-8fd7-c2725d718538\", \"YOLO11.zip\", 235219091)"
            ]
          },
          "metadata": {}
        }
      ]
    }
  ]
}